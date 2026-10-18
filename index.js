/**
 * Failed Vector Purger
 *
 * 標準の Vector Storage 拡張機能でベクトル化が最後まで完了しなかったファイル
 * （Data Bank / チャット添付ファイル）のコレクションだけを検出して削除する。
 *
 * 標準機能はファイルのコレクションに 1 件でもハッシュが存在すると
 * 「ベクトル化済み」と判断して再処理しないため、途中で失敗したファイルは
 * 不完全なまま放置される。本拡張機能はそれらを空に戻すことで、
 * 標準機能の「Vectorize All」で不足分のファイルのみを再ベクトル化できるようにする。
 *
 * 失敗が繰り返される環境向けに、標準機能と同じチャンク・同じ API で
 * 不足チャンクだけを 1 件ずつ再試行しながら追加する「補完」モードも持つ。
 */

import { getRequestHeaders, saveSettingsDebounced, eventSource, event_types } from '../../../../script.js';
import { extension_settings, getContext } from '../../../extensions.js';
import { getStringHash, splitRecursive, trimToStartSentence, trimToEndSentence, escapeHtml } from '../../../utils.js';
import { getDataBankAttachments, getFileAttachment } from '../../../chats.js';
import { textgen_types, textgenerationwebui_settings } from '../../../textgen-settings.js';
import { oai_settings } from '../../../openai.js';
import { SlashCommandParser } from '../../../slash-commands/SlashCommandParser.js';
import { SlashCommand } from '../../../slash-commands/SlashCommand.js';
import { ARGUMENT_TYPE, SlashCommandNamedArgument } from '../../../slash-commands/SlashCommandArgument.js';

const MODULE_NAME = 'failed_vector_purger';
const LOG_PREFIX = 'Failed Vector Purger:';
const VERSION = '1.2.0';

const defaultSettings = {
    include_disabled: true,
    purge_mismatch: false,
    auto_revectorize: false,
    auto_purge_on_chat_change: false,
    fill_retries: 3,
};

/**
 * 判定結果
 * - complete: すべてのチャンクが保存されている
 * - incomplete: 保存済みチャンクが期待値の部分集合で、不足がある（＝ベクトル化失敗）
 * - mismatch: 期待しないチャンクが含まれている（チャンク設定変更・翻訳などで照合不能）
 * - empty: コレクションが空（失敗して 1 チャンクも保存されなかった or 未処理）。
 *   削除は不要だが再ベクトル化が必要なので、要ベクトル化として集計する
 * - error: 判定できなかった
 */
const STATUS = Object.freeze({
    COMPLETE: 'complete',
    INCOMPLETE: 'incomplete',
    MISMATCH: 'mismatch',
    EMPTY: 'empty',
    ERROR: 'error',
});

const STATUS_LABELS = {
    [STATUS.COMPLETE]: '完了',
    [STATUS.INCOMPLETE]: '不完全',
    [STATUS.MISMATCH]: '設定不一致',
    [STATUS.EMPTY]: '未ベクトル化',
    [STATUS.ERROR]: 'エラー',
};

/**
 * @typedef {object} ScanResult
 * @property {string} url ファイル URL
 * @property {string} name ファイル名
 * @property {string[]} origins 出所（Data Bank / Chat）
 * @property {string} collectionId コレクション ID
 * @property {string} status 判定結果
 * @property {number} saved 保存済みのユニークチャンク数
 * @property {number} expected 期待されるユニークチャンク数（最も近い候補）
 * @property {boolean} disabled 無効化された Data Bank ファイルか（標準の Vectorize All は処理しない）
 * @property {string} [message] 補足
 */

/** @type {ScanResult[]} */
let lastResults = [];
let busy = false;
let koboldModelCache = null;
let cancelRequested = false;

function getSettings() {
    extension_settings[MODULE_NAME] = Object.assign({}, defaultSettings, extension_settings[MODULE_NAME]);
    return extension_settings[MODULE_NAME];
}

/**
 * 標準 Vector Storage の設定を取得する。
 * @returns {object|null}
 */
function getVectorSettings() {
    return extension_settings.vectors ?? null;
}

/**
 * 標準機能と同じ方法でファイルのコレクション ID を求める。
 * @param {string} fileUrl
 */
function getFileCollectionId(fileUrl) {
    return `file_${getStringHash(fileUrl)}`;
}

/**
 * 保存先のパス（source / collectionId / model）を一致させるため、
 * 標準機能の getVectorsRequestBody と同じ model 指定を組み立てる。
 * @param {object} vs 標準 Vector Storage の設定
 * @returns {Promise<object>}
 */
async function getListRequestBody(vs) {
    const body = { source: vs.source };
    const modelKeys = {
        electronhub: 'electronhub_model',
        openrouter: 'openrouter_model',
        togetherai: 'togetherai_model',
        openai: 'openai_model',
        cohere: 'cohere_model',
        ollama: 'ollama_model',
        vllm: 'vllm_model',
        webllm: 'webllm_model',
        palm: 'google_model',
        vertexai: 'google_model',
        chutes: 'chutes_model',
        nanogpt: 'nanogpt_model',
        siliconflow: 'siliconflow_model',
    };

    if (modelKeys[vs.source]) {
        body.model = vs[modelKeys[vs.source]];
    }

    switch (vs.source) {
        case 'palm':
            body.api = 'makersuite';
            break;
        case 'vertexai':
            body.api = 'vertexai';
            break;
        case 'workers_ai':
            body.model = vs.workers_ai_model || '@cf/baai/bge-m3';
            break;
        case 'koboldcpp':
            body.model = await getKoboldCppModel(vs);
            break;
    }

    return body;
}

/**
 * KoboldCpp はモデル名をサーバーから取得する必要がある。
 * @param {object} vs
 * @returns {Promise<string>}
 */
async function getKoboldCppModel(vs) {
    if (koboldModelCache) {
        return koboldModelCache;
    }

    const response = await fetch('/api/backends/kobold/embed', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            items: ['model probe'],
            server: vs.use_alt_endpoint ? vs.alt_endpoint_url : textgenerationwebui_settings.server_urls[textgen_types.KOBOLDCPP],
        }),
    });

    if (!response.ok) {
        throw new Error('KoboldCpp からモデル名を取得できませんでした');
    }

    const data = await response.json();
    koboldModelCache = String(data.model || '');
    return koboldModelCache;
}

/**
 * コレクションに保存されているハッシュ一覧を取得する。
 * @param {string} collectionId
 * @param {object} listBody
 * @returns {Promise<number[]>}
 */
async function getSavedHashes(collectionId, listBody) {
    const response = await fetch('/api/vector/list', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ ...listBody, collectionId }),
    });

    if (!response.ok) {
        throw new Error(`コレクション ${collectionId} のハッシュ取得に失敗しました`);
    }

    return (await response.json()).map(Number);
}

/**
 * コレクションを削除する（標準機能の purge と同じエンドポイント）。
 * @param {string} collectionId
 */
async function purgeCollection(collectionId) {
    const response = await fetch('/api/vector/purge', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ collectionId }),
    });

    if (!response.ok) {
        throw new Error(`コレクション ${collectionId} の削除に失敗しました`);
    }
}

function getChunkDelimiters(vs) {
    const delimiters = ['\n\n', '\n', ' ', ''];

    if (vs.force_chunk_delimiter) {
        delimiters.unshift(vs.force_chunk_delimiter);
    }

    return delimiters;
}

function overlapChunks(chunk, index, chunks, overlapSize) {
    const halfOverlap = Math.floor(overlapSize / 2);
    const nextChunk = chunks[index + 1];
    const prevChunk = chunks[index - 1];

    const nextOverlap = trimToEndSentence(nextChunk?.substring(0, halfOverlap)) || '';
    const prevOverlap = trimToStartSentence(prevChunk?.substring(prevChunk.length - halfOverlap)) || '';
    return [prevOverlap, chunk, nextOverlap].filter(x => x).join(' ');
}

/**
 * 標準機能の vectorizeFile と同じ手順でチャンクのハッシュ集合を求める。
 * @param {string} fileText
 * @param {number} chunkSize
 * @param {number} overlapPercent
 * @param {object} vs
 * @returns {Set<number>}
 */
function computeExpectedHashes(fileText, chunkSize, overlapPercent, vs) {
    return new Set(computeChunkItems(fileText, chunkSize, overlapPercent, vs).map(item => item.hash));
}

/**
 * 標準機能の vectorizeFile と同じ手順でチャンクを作る。
 * @returns {{hash: number, text: string, index: number}[]}
 */
function computeChunkItems(fileText, chunkSize, overlapPercent, vs) {
    const overlapSize = Math.round(chunkSize * overlapPercent / 100);
    const delimiters = getChunkDelimiters(vs);
    chunkSize = overlapSize > 0 ? (chunkSize - overlapSize) : chunkSize;
    const applyOverlap = (x, y, z) => overlapSize > 0 ? overlapChunks(x, y, z, overlapSize) : x;
    const chunks = vs.only_custom_boundary && vs.force_chunk_delimiter
        ? fileText.split(vs.force_chunk_delimiter).map(applyOverlap)
        : splitRecursive(fileText, chunkSize, delimiters).map(applyOverlap);

    return chunks.map((chunk, index) => ({ hash: getStringHash(chunk), text: chunk, index }));
}

/**
 * 検査対象のファイルを収集する。標準機能の取り込み経路ごとにチャンク設定の候補を持たせる。
 * @param {object} vs
 * @returns {Map<string, {url: string, name: string, origins: Set<string>, file: object, variants: {chunkSize: number, overlap: number}[]}>}
 */
function collectTargets(vs) {
    const settings = getSettings();
    const targets = new Map();

    const add = (file, origin, variants) => {
        if (!file?.url) {
            return;
        }

        const entry = targets.get(file.url) ?? { url: file.url, name: file.name || file.url, origins: new Set(), file, variants: [], disabled: false };
        entry.origins.add(origin);

        for (const variant of variants) {
            if (!entry.variants.some(v => v.chunkSize === variant.chunkSize && v.overlap === variant.overlap)) {
                entry.variants.push(variant);
            }
        }

        targets.set(file.url, entry);
    };

    // Data Bank: ingestDataBankAttachments / Vectorize All と同じ規則
    const thresholdDb = Number(vs.size_threshold_db) * 1024;
    const enabledUrls = new Set(getDataBankAttachments(false).map(x => x.url));
    for (const file of getDataBankAttachments(settings.include_disabled)) {
        const isNew = !targets.has(file.url);
        add(file, 'Data Bank', [{
            chunkSize: file.size > thresholdDb ? Number(vs.chunk_size_db) : -1,
            overlap: Number(vs.overlap_percent_db),
            preferred: true,
        }]);
        if (isNew && !enabledUrls.has(file.url)) {
            targets.get(file.url).disabled = true;
        }
    }

    // チャット添付: processFiles は常に chunk_size、Vectorize All はしきい値で分岐するので両方を候補にする
    const threshold = Number(vs.size_threshold) * 1024;
    const chat = getContext().chat ?? [];
    for (const message of chat) {
        if (!Array.isArray(message?.extra?.files)) {
            continue;
        }

        for (const file of message.extra.files) {
            // チャット添付は無効化の概念がなく、Vectorize All の対象になる
            if (targets.has(file.url)) {
                targets.get(file.url).disabled = false;
            }
            add(file, 'Chat', [
                { chunkSize: Number(vs.chunk_size), overlap: Number(vs.overlap_percent) },
                // Vectorize All と同じ規則。空のコレクションを補完するときはこちらを使う
                { chunkSize: file.size > threshold ? Number(vs.chunk_size) : -1, overlap: Number(vs.overlap_percent), preferred: true },
            ]);
        }
    }

    return targets;
}

/**
 * 1 ファイル分を判定する。
 * @returns {Promise<ScanResult>}
 */
async function inspectTarget(target, vs, listBody) {
    const collectionId = getFileCollectionId(target.url);
    const result = {
        url: target.url,
        name: target.name,
        origins: [...target.origins],
        collectionId,
        status: STATUS.ERROR,
        saved: 0,
        expected: 0,
        disabled: target.disabled,
    };

    // 補完モード用の情報（表示には使わない）
    Object.defineProperty(result, 'plan', { value: { target, savedSet: new Set(), text: null, variant: null }, enumerable: false });

    try {
        const savedSet = new Set(await getSavedHashes(collectionId, listBody));
        result.saved = savedSet.size;
        result.plan.savedSet = savedSet;

        if (savedSet.size === 0) {
            result.status = STATUS.EMPTY;
            result.message = result.disabled
                ? '無効化中のため標準機能は処理しません'
                : '要ベクトル化（失敗して 1 チャンクも保存されていないか、未処理です）';
            return result;
        }

        // 標準機能は経路によって file.text または取得したファイル本文を使う
        const texts = new Set();
        if (typeof target.file.text === 'string' && target.file.text) {
            texts.add(target.file.text);
        }
        const fetched = await getFileAttachment(target.url);
        if (typeof fetched === 'string' && fetched) {
            texts.add(fetched);
        }

        if (texts.size === 0) {
            result.message = 'ファイル本文を取得できませんでした';
            return result;
        }

        let best = null;
        for (const text of texts) {
            for (const variant of target.variants) {
                const expected = computeExpectedHashes(text, variant.chunkSize, variant.overlap, vs);
                const missing = [...expected].filter(h => !savedSet.has(h)).length;
                const extra = [...savedSet].filter(h => !expected.has(h)).length;
                const candidate = { expected: expected.size, missing, extra, text, variant };

                if (!best || extra < best.extra || (extra === best.extra && missing < best.missing)) {
                    best = candidate;
                }
            }
        }

        result.expected = best.expected;
        result.plan.text = best.text;
        result.plan.variant = best.variant;

        if (best.missing === 0 && best.extra === 0) {
            result.status = STATUS.COMPLETE;
        } else if (best.extra === 0) {
            result.status = STATUS.INCOMPLETE;
            result.message = `${best.missing} チャンク不足`;
        } else if (best.missing === 0) {
            // 期待チャンクはすべて揃っている（古い余剰チャンクのみ）
            result.status = STATUS.COMPLETE;
            result.message = `余剰 ${best.extra} チャンク`;
        } else {
            result.status = STATUS.MISMATCH;
            result.message = vs.translate_files
                ? '翻訳設定が有効なため照合できません'
                : `照合不能（不足 ${best.missing} / 余剰 ${best.extra}）。チャンク設定が変更された可能性があります`;
        }
    } catch (error) {
        console.error(LOG_PREFIX, `Failed to inspect ${target.url}`, error);
        result.message = String(error?.message || error);
    }

    return result;
}

/**
 * 再ベクトル化が必要か（空のコレクション。削除済みのものも含む）。
 */
function needsVectorize(result) {
    return result.status === STATUS.EMPTY && !result.disabled;
}

/**
 * 検査結果の件数を集計して表示用の文字列にする。
 * @param {ScanResult[]} results
 */
function summarize(results) {
    const count = status => results.filter(r => r.status === status).length;
    return `完了 ${count(STATUS.COMPLETE)} / 不完全 ${count(STATUS.INCOMPLETE)} / 未ベクトル化 ${results.filter(needsVectorize).length}`
        + ` / 設定不一致 ${count(STATUS.MISMATCH)} / エラー ${count(STATUS.ERROR)}`;
}

function isPurgeTarget(result) {
    const settings = getSettings();
    return result.status === STATUS.INCOMPLETE || (settings.purge_mismatch && result.status === STATUS.MISMATCH);
}

/**
 * 現在のコンテキストにある全ファイルを検査する。
 * @returns {Promise<ScanResult[]>}
 */
async function scan() {
    const vs = getVectorSettings();
    if (!vs) {
        throw new Error('標準の Vector Storage 拡張機能が見つかりません');
    }

    const listBody = await getListRequestBody(vs);
    const targets = collectTargets(vs);
    const results = [];

    for (const target of targets.values()) {
        results.push(await inspectTarget(target, vs, listBody));
    }

    lastResults = results;
    return results;
}

/**
 * 指定したコレクションを削除する。
 * @param {ScanResult[]} results
 * @returns {Promise<number>} 削除した件数
 */
async function purge(results) {
    let count = 0;

    for (const result of results) {
        try {
            await purgeCollection(result.collectionId);
            console.log(LOG_PREFIX, `Purged ${result.collectionId} (${result.name})`);
            result.status = STATUS.EMPTY;
            result.saved = 0;
            result.message = '削除済み（要ベクトル化）';
            count++;
        } catch (error) {
            console.error(LOG_PREFIX, `Failed to purge ${result.collectionId}`, error);
            result.message = String(error?.message || error);
        }
    }

    return count;
}

/**
 * 標準 Vector Storage の「Vectorize All（ファイル）」を実行する。
 * 空になったコレクションのファイルだけが処理される。
 */
function triggerRevectorize() {
    const button = $('#vectors_files_vectorize_all');
    if (!button.length) {
        toastr.warning('標準 Vector Storage の「Vectorize All」ボタンが見つかりません', 'Failed Vector Purger');
        return false;
    }

    button.trigger('click');
    return true;
}

// ---------------------------------------------------------------------------
// 補完モード: 不足チャンクだけを 1 件ずつ再試行しながら追加する
// ---------------------------------------------------------------------------

/**
 * 標準機能の getVectorsRequestBody と同じ内容で insert 用のリクエストボディを作る。
 * @param {object} vs
 * @param {string[]} texts 埋め込み対象（KoboldCpp ではクライアント側で埋め込みを作る）
 * @returns {Promise<object>}
 */
async function getInsertRequestBody(vs, texts) {
    const body = await getListRequestBody(vs);
    const serverUrl = type => vs.use_alt_endpoint ? vs.alt_endpoint_url : textgenerationwebui_settings.server_urls[type];

    switch (vs.source) {
        case 'extras':
            body.extrasUrl = extension_settings.apiUrl;
            body.extrasKey = extension_settings.apiKey;
            break;
        case 'ollama':
            body.apiUrl = serverUrl(textgen_types.OLLAMA);
            body.keep = !!vs.ollama_keep;
            break;
        case 'llamacpp':
            body.apiUrl = serverUrl(textgen_types.LLAMACPP);
            break;
        case 'vllm':
            body.apiUrl = serverUrl(textgen_types.VLLM);
            break;
        case 'vertexai':
            body.vertexai_auth_mode = oai_settings.vertexai_auth_mode;
            body.vertexai_region = oai_settings.vertexai_region;
            body.vertexai_express_project_id = oai_settings.vertexai_express_project_id;
            break;
        case 'siliconflow':
            body.siliconflow_endpoint = oai_settings.siliconflow_endpoint;
            break;
        case 'workers_ai':
            body.workers_ai_account_id = oai_settings.workers_ai_account_id;
            break;
        case 'koboldcpp': {
            const response = await fetch('/api/backends/kobold/embed', {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({ items: texts, server: serverUrl(textgen_types.KOBOLDCPP) }),
            });
            if (!response.ok) {
                throw new Error('KoboldCpp の埋め込み取得に失敗しました');
            }
            const data = await response.json();
            if (!Array.isArray(data.embeddings) || data.embeddings.length !== texts.length || data.embeddings.some(e => !Array.isArray(e) || !e.length)) {
                throw new Error('KoboldCpp から不正な埋め込みが返されました');
            }
            body.model = data.model;
            body.embeddings = Object.fromEntries(texts.map((t, i) => [t, data.embeddings[i]]));
            break;
        }
    }

    return body;
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/**
 * 1 チャンクを挿入する。失敗したら指数バックオフで再試行する。
 * @returns {Promise<string|null>} 最後のエラー内容（成功時は null）
 */
async function insertWithRetry(collectionId, item, vs, retries) {
    let lastError = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
        if (cancelRequested) {
            return '中止されました';
        }

        if (attempt > 0) {
            await sleep(2000 * 2 ** (attempt - 1));
        }

        try {
            const body = await getInsertRequestBody(vs, [item.text]);
            const response = await fetch('/api/vector/insert', {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({ ...body, collectionId, items: [item] }),
            });

            if (response.ok) {
                return null;
            }

            lastError = `HTTP ${response.status}`;
        } catch (error) {
            lastError = String(error?.message || error);
        }

        console.warn(LOG_PREFIX, `Insert failed for ${collectionId} chunk #${item.index} (attempt ${attempt + 1}/${retries + 1}): ${lastError}`);
    }

    return lastError;
}

/**
 * 1 ファイル分の不足チャンクを補完する。
 * @param {ScanResult} result
 * @param {object} vs
 * @param {(done: number, total: number) => void} onProgress
 * @returns {Promise<{inserted: number, failed: {index: number, length: number, error: string}[]}>}
 */
async function fillMissing(result, vs, onProgress) {
    if (vs.translate_files) {
        throw new Error('Vector Storage の「Translate files」が有効なため補完できません');
    }

    if (vs.source === 'webllm') {
        throw new Error('WebLLM は補完モードに対応していません');
    }

    const plan = result.plan;
    let { text, variant } = plan;

    if (!text || !variant) {
        // 空のコレクション: 標準の Vectorize All と同じ規則でチャンクを作る
        text = (typeof plan.target.file.text === 'string' && plan.target.file.text) || await getFileAttachment(plan.target.url);
        variant = plan.target.variants.find(v => v.preferred) ?? plan.target.variants[0];
    }

    if (!text) {
        throw new Error('ファイル本文を取得できませんでした');
    }

    const items = computeChunkItems(text, variant.chunkSize, variant.overlap, vs).filter(item => !plan.savedSet.has(item.hash));
    const retries = Math.max(0, Number(getSettings().fill_retries) || 0);
    const failed = [];
    let inserted = 0;

    for (let i = 0; i < items.length; i++) {
        if (cancelRequested) {
            break;
        }

        onProgress(i, items.length);
        const error = await insertWithRetry(result.collectionId, items[i], vs, retries);

        if (error) {
            failed.push({ index: items[i].index, length: items[i].text.length, error });
            console.error(LOG_PREFIX, `Chunk #${items[i].index} of ${result.name} (${items[i].text.length} chars) failed: ${error}`, items[i].text);
        } else {
            inserted++;
            plan.savedSet.add(items[i].hash);
        }
    }

    onProgress(items.length, items.length);
    return { inserted, failed };
}

/**
 * 検査して、不完全・未ベクトル化のファイルを補完する。
 * @returns {Promise<{files: number, inserted: number, failed: number}>}
 */
async function scanAndFill({ silent = false } = {}) {
    if (busy) {
        throw new Error('処理中です');
    }

    if (isVectorizationRunning()) {
        throw new Error('標準 Vector Storage がファイルを取り込み中です。完了してから実行してください。');
    }

    busy = true;
    cancelRequested = false;
    setBusyUi(true);
    $('#fvp_cancel').show();

    const toastBody = $('<span>').text('検査中...');
    const toast = toastr.info(toastBody, '不足チャンクを補完中', { timeOut: 0, extendedTimeOut: 0, closeButton: false, escapeHtml: false });
    let inserted = 0;
    let failed = 0;
    let files = 0;

    try {
        const vs = getVectorSettings();
        const results = await scan();
        const targets = results.filter(r => r.status === STATUS.INCOMPLETE || needsVectorize(r));

        for (let i = 0; i < targets.length && !cancelRequested; i++) {
            const result = targets[i];
            files++;

            try {
                const outcome = await fillMissing(result, vs, (done, total) => {
                    toastBody.text(`ファイル ${i + 1}/${targets.length}: ${result.name} — ${done}/${total} チャンク`);
                });
                inserted += outcome.inserted;
                failed += outcome.failed.length;

                if (outcome.failed.length) {
                    const list = outcome.failed.slice(0, 5).map(f => `#${f.index}（${f.length} 文字）: ${f.error}`).join(' / ');
                    result.message = `${outcome.failed.length} チャンクが再試行後も失敗: ${list}`;
                }
            } catch (error) {
                failed++;
                result.message = String(error?.message || error);
                console.error(LOG_PREFIX, `Fill failed for ${result.name}`, error);
            }
        }

        // 結果を検査し直して表示する（失敗メッセージは引き継ぐ）
        const messages = new Map(results.filter(r => r.message).map(r => [r.collectionId, r.message]));
        const rescanned = await scan();
        for (const r of rescanned) {
            if (r.status !== STATUS.COMPLETE && messages.has(r.collectionId)) {
                r.message = messages.get(r.collectionId);
            }
        }
        renderResults();

        if (!silent) {
            const summary = summarize(rescanned);
            const head = cancelRequested ? '中止しました。' : '補完が終わりました。';
            const type = failed || cancelRequested ? 'warning' : 'success';
            toastr[type](`${head}追加 ${inserted} チャンク / 失敗 ${failed}<br>${summary}`, 'Failed Vector Purger', { escapeHtml: false, timeOut: 10000 });
        }

        return { files, inserted, failed };
    } finally {
        toastr.clear(toast);
        busy = false;
        setBusyUi(false);
        $('#fvp_cancel').hide();
    }
}

/**
 * 検査して不完全なものを削除する一連の処理。
 * @param {{dryRun?: boolean, silent?: boolean, revectorize?: boolean}} options
 * @returns {Promise<{scanned: number, purged: number, targets: number, pending: number}>}
 */
async function scanAndPurge({ dryRun = false, silent = false, revectorize = getSettings().auto_revectorize } = {}) {
    if (busy) {
        throw new Error('処理中です');
    }

    if (isVectorizationRunning()) {
        throw new Error('標準 Vector Storage がファイルを取り込み中です。完了してから実行してください。');
    }

    busy = true;
    setBusyUi(true);

    try {
        const results = await scan();
        const targets = results.filter(isPurgeTarget);
        let purged = 0;

        if (!dryRun && targets.length) {
            purged = await purge(targets);
        }

        renderResults();

        // 削除したものに加え、最初から空のもの（1 バッチ目で失敗したもの）も再ベクトル化が必要
        const pending = results.filter(needsVectorize).length;
        const willRevectorize = !dryRun && pending > 0 && revectorize;

        if (!silent) {
            const summary = summarize(results);
            if (!targets.length && !pending) {
                toastr.success(`すべてのファイルのベクトル化が完了しています。<br>${summary}`, 'Failed Vector Purger', { escapeHtml: false });
            } else if (dryRun) {
                toastr.warning(`削除候補 ${targets.length} 件 / 要ベクトル化 ${pending} 件<br>${summary}`, 'Failed Vector Purger', { escapeHtml: false });
            } else {
                const next = willRevectorize ? '再ベクトル化を開始します。' : '「再ベクトル化」を実行してください。';
                toastr.warning(`${purged} 件を削除しました。要ベクトル化 ${pending} 件。${next}<br>${summary}`, 'Failed Vector Purger', { escapeHtml: false });
            }
        }

        if (willRevectorize) {
            triggerRevectorize();
        }

        return { scanned: results.length, targets: targets.length, purged, pending };
    } finally {
        busy = false;
        setBusyUi(false);
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

function setBusyUi(isBusy) {
    $('#fvp_container .menu_button').not('#fvp_cancel').toggleClass('disabled', isBusy);
    $('#fvp_spinner').toggle(isBusy);
}

function renderResults() {
    const container = $('#fvp_results');
    container.empty();
    $('#fvp_summary').text(lastResults.length ? summarize(lastResults) : '');

    if (!lastResults.length) {
        container.append('<div class="fvp_empty">検査対象のファイルはありません。</div>');
        return;
    }

    const order = [STATUS.INCOMPLETE, STATUS.EMPTY, STATUS.MISMATCH, STATUS.ERROR, STATUS.COMPLETE];
    const sorted = [...lastResults].sort((a, b) => order.indexOf(a.status) - order.indexOf(b.status));

    for (const result of sorted) {
        const selectable = result.status === STATUS.INCOMPLETE || result.status === STATUS.MISMATCH;
        const checked = isPurgeTarget(result) ? 'checked' : '';
        const progress = result.expected ? `${result.saved} / ${result.expected}` : `${result.saved}`;
        const row = $(`
            <div class="fvp_row fvp_status_${result.status} ${needsVectorize(result) ? 'fvp_needs_vectorize' : ''}">
                <input type="checkbox" class="fvp_select" ${selectable ? '' : 'disabled'} ${selectable ? checked : ''}>
                <span class="fvp_badge">${STATUS_LABELS[result.status]}</span>
                <span class="fvp_name" title="${escapeHtml(result.url)}">${escapeHtml(result.name)}</span>
                <span class="fvp_origin">${escapeHtml(result.origins.join(', '))}</span>
                <span class="fvp_progress" title="保存済み / 期待チャンク数">${progress}</span>
                ${result.message ? `<div class="fvp_message">${escapeHtml(result.message)}</div>` : ''}
            </div>`);
        row.data('collectionId', result.collectionId);
        container.append(row);
    }
}

/**
 * 標準機能がファイルを取り込み中かどうか（取り込み中のコレクションを誤って削除しないため）。
 */
function isVectorizationRunning() {
    return $('#toast-container .toast-title').filter((_, el) => $(el).text().startsWith('Ingesting file')).length > 0;
}

async function onScanClick() {
    try {
        await scanAndPurge({ dryRun: true });
    } catch (error) {
        console.error(LOG_PREFIX, error);
        toastr.error(String(error?.message || error), 'Failed Vector Purger');
    }
}

async function onPurgeSelectedClick() {
    if (busy) return;

    const selectedIds = $('#fvp_results .fvp_row').filter((_, el) => $(el).find('.fvp_select').prop('checked'))
        .map((_, el) => $(el).data('collectionId')).get();
    const selected = lastResults.filter(r => selectedIds.includes(r.collectionId));

    if (isVectorizationRunning()) {
        toastr.warning('標準 Vector Storage がファイルを取り込み中です。完了してから実行してください。', 'Failed Vector Purger');
        return;
    }

    if (!selected.length) {
        toastr.info('削除対象が選択されていません。先に「検査」を実行してください。', 'Failed Vector Purger');
        return;
    }

    busy = true;
    setBusyUi(true);

    try {
        const purged = await purge(selected);
        renderResults();
        toastr.success(`${purged} 件のコレクションを削除しました`, 'Failed Vector Purger');

        if (lastResults.some(needsVectorize) && getSettings().auto_revectorize) {
            triggerRevectorize();
        }
    } catch (error) {
        console.error(LOG_PREFIX, error);
        toastr.error(String(error?.message || error), 'Failed Vector Purger');
    } finally {
        busy = false;
        setBusyUi(false);
    }
}

async function onScanAndPurgeClick() {
    try {
        await scanAndPurge();
    } catch (error) {
        console.error(LOG_PREFIX, error);
        toastr.error(String(error?.message || error), 'Failed Vector Purger');
    }
}

async function onFillClick() {
    try {
        await scanAndFill();
    } catch (error) {
        console.error(LOG_PREFIX, error);
        toastr.error(String(error?.message || error), 'Failed Vector Purger');
    }
}

async function onChatChanged() {
    if (!getSettings().auto_purge_on_chat_change || busy || isVectorizationRunning()) {
        return;
    }

    try {
        const { purged, pending } = await scanAndPurge({ silent: true });
        if (purged > 0 || pending > 0) {
            toastr.info(`不完全なベクトル ${purged} 件を自動削除しました。要ベクトル化 ${pending} 件`, 'Failed Vector Purger');
        }
    } catch (error) {
        console.error(LOG_PREFIX, 'Auto purge failed', error);
    }
}

function getSettingsHtml() {
    return `
    <div id="fvp_container" class="fvp_container">
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>Failed Vector Purger <small>v${VERSION}</small></b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
                <small class="fvp_desc">
                    標準 Vector Storage でベクトル化が途中で失敗したファイル（Data Bank・チャット添付）のコレクションだけを削除します。
                    削除後に標準機能の「Vectorize All」を実行すると、削除したファイルだけが再ベクトル化されます。
                </small>
                <label class="checkbox_label" for="fvp_include_disabled">
                    <input type="checkbox" id="fvp_include_disabled">
                    <span>無効化された Data Bank ファイルも検査する</span>
                </label>
                <label class="checkbox_label" for="fvp_purge_mismatch">
                    <input type="checkbox" id="fvp_purge_mismatch">
                    <span>「設定不一致」も削除対象にする（チャンク設定変更後の作り直し用）</span>
                </label>
                <label class="checkbox_label" for="fvp_auto_revectorize">
                    <input type="checkbox" id="fvp_auto_revectorize">
                    <span>削除後に標準機能の「Vectorize All」を自動実行する</span>
                </label>
                <label class="checkbox_label" for="fvp_auto_purge_on_chat_change">
                    <input type="checkbox" id="fvp_auto_purge_on_chat_change">
                    <span>チャット切り替え時に自動で検査・削除する</span>
                </label>
                <label for="fvp_fill_retries">
                    <small>補完時の再試行回数（1 チャンクあたり）</small>
                    <input type="number" id="fvp_fill_retries" class="text_pole" min="0" max="10" step="1">
                </label>
                <div class="flex-container">
                    <div id="fvp_scan" class="menu_button menu_button_icon" title="削除せずに検査だけ行います">
                        <i class="fa-solid fa-magnifying-glass"></i><span>検査</span>
                    </div>
                    <div id="fvp_purge_selected" class="menu_button menu_button_icon" title="チェックしたコレクションを削除します">
                        <i class="fa-solid fa-trash"></i><span>選択を削除</span>
                    </div>
                    <div id="fvp_scan_purge" class="menu_button menu_button_icon" title="検査して不完全なものをすべて削除します">
                        <i class="fa-solid fa-broom"></i><span>検査して削除</span>
                    </div>
                    <div id="fvp_revectorize" class="menu_button menu_button_icon" title="標準 Vector Storage の Vectorize All（ファイル）を実行します">
                        <i class="fa-solid fa-rotate"></i><span>再ベクトル化</span>
                    </div>
                    <div id="fvp_fill" class="menu_button menu_button_icon" title="削除せず、不足しているチャンクだけを 1 件ずつ再試行しながら追加します">
                        <i class="fa-solid fa-puzzle-piece"></i><span>不足分を補完</span>
                    </div>
                    <div id="fvp_cancel" class="menu_button menu_button_icon" style="display:none" title="補完を中止します">
                        <i class="fa-solid fa-stop"></i><span>中止</span>
                    </div>
                    <i id="fvp_spinner" class="fa-solid fa-spinner fa-spin" style="display:none"></i>
                </div>
                <div id="fvp_summary" class="fvp_summary"></div>
                <div id="fvp_results" class="fvp_results"></div>
            </div>
        </div>
    </div>`;
}

function bindCheckbox(id, key) {
    const settings = getSettings();
    $(`#${id}`).prop('checked', !!settings[key]).on('input', function () {
        getSettings()[key] = !!$(this).prop('checked');
        saveSettingsDebounced();
        if (key === 'purge_mismatch') {
            renderResults();
        }
    });
}

function registerSlashCommand() {
    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-purge-failed',
        callback: async (args) => {
            const dryRun = String(args.dry ?? 'false').toLowerCase() === 'true';
            const revectorize = args.revectorize === undefined
                ? getSettings().auto_revectorize
                : String(args.revectorize).toLowerCase() === 'true';
            const silent = String(args.quiet ?? 'false').toLowerCase() === 'true';
            const { targets, purged } = await scanAndPurge({ dryRun, revectorize, silent });
            return String(dryRun ? targets : purged);
        },
        namedArgumentList: [
            SlashCommandNamedArgument.fromProps({
                name: 'dry',
                description: 'true の場合は検査のみ行い、削除候補の件数を返す',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
                defaultValue: 'false',
            }),
            SlashCommandNamedArgument.fromProps({
                name: 'revectorize',
                description: '削除後に標準機能の Vectorize All を実行するか（省略時は設定に従う）',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
            }),
            SlashCommandNamedArgument.fromProps({
                name: 'quiet',
                description: 'true の場合は通知を表示しない',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
                defaultValue: 'false',
            }),
        ],
        helpString: 'ベクトル化が完了しなかったファイルのベクトルコレクションを削除し、削除件数を返します。',
        returns: '削除した件数（dry=true の場合は削除候補の件数）',
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-fill-missing',
        callback: async (args) => {
            const silent = String(args.quiet ?? 'false').toLowerCase() === 'true';
            const { inserted } = await scanAndFill({ silent });
            return String(inserted);
        },
        namedArgumentList: [
            SlashCommandNamedArgument.fromProps({
                name: 'quiet',
                description: 'true の場合は完了通知を表示しない',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
                defaultValue: 'false',
            }),
        ],
        helpString: '不完全・未ベクトル化のファイルについて、不足しているチャンクだけを 1 件ずつ再試行しながら追加します。',
        returns: '追加したチャンク数',
    }));
}

jQuery(async () => {
    getSettings();
    console.log(LOG_PREFIX, `v${VERSION} loaded`);

    const html = getSettingsHtml();
    const target = $('#extensions_settings2').length ? $('#extensions_settings2') : $('#extensions_settings');
    target.append(html);

    bindCheckbox('fvp_include_disabled', 'include_disabled');
    bindCheckbox('fvp_purge_mismatch', 'purge_mismatch');
    bindCheckbox('fvp_auto_revectorize', 'auto_revectorize');
    bindCheckbox('fvp_auto_purge_on_chat_change', 'auto_purge_on_chat_change');

    $('#fvp_scan').on('click', onScanClick);
    $('#fvp_purge_selected').on('click', onPurgeSelectedClick);
    $('#fvp_scan_purge').on('click', onScanAndPurgeClick);
    $('#fvp_revectorize').on('click', () => triggerRevectorize());
    $('#fvp_fill').on('click', onFillClick);
    $('#fvp_cancel').on('click', () => {
        cancelRequested = true;
    });
    $('#fvp_fill_retries').val(getSettings().fill_retries).on('input', function () {
        getSettings().fill_retries = Math.max(0, Math.min(10, Number($(this).val()) || 0));
        saveSettingsDebounced();
    });

    eventSource.on(event_types.CHAT_CHANGED, onChatChanged);

    try {
        registerSlashCommand();
    } catch (error) {
        console.error(LOG_PREFIX, 'Failed to register slash command', error);
    }
});
