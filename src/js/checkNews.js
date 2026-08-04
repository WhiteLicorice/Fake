const API_ORIGIN = "https://fph-ml.onrender.com";
const API_ENDPOINT = `${API_ORIGIN}/check-news`;
const REQUEST_TIMEOUT_MS = 150000;
const COLD_START_NOTICE_MS = 8000;

const button = document.getElementById('check_button');
const statusElement = document.getElementById('analysis_status');

// Begin waking the free Render service while the reader is choosing an article.
// A failed warm-up is harmless because the prediction request retries the service.
fetch(`${API_ORIGIN}/`).catch(() => {});

button.addEventListener('click', async () => {
    const content = document.getElementById('content_input').value.trim();
    const resultContainer = document.getElementById('result_container');

    resultContainer.className = "hidden bg-slate-50 border-t border-slate-100 p-6 md:p-8 animate-fade-in";

    if (content.length < 20) {
        displayError('Paste at least 20 characters of Filipino news text before analyzing.');
        return;
    }

    const originalButtonText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Analyzing...';
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    setStatus('Connecting to the prediction service...');

    const coldStartNotice = window.setTimeout(() => {
        setStatus('The prediction service is waking up. Keep this page open; the first request may take up to two minutes.');
    }, COLD_START_NOTICE_MS);

    try {
        const isFake = await checkIfFakeNews(content);
        displayResult(isFake);
    } catch (error) {
        if (error.name === 'AbortError') {
            displayError('The prediction service did not respond within 150 seconds. Please try again.');
        } else {
            displayError(error.message);
        }
    } finally {
        window.clearTimeout(coldStartNotice);
        hideStatus();
        button.innerHTML = originalButtonText;
        button.disabled = false;
        button.removeAttribute('aria-busy');
    }
});

function setStatus(message) {
    statusElement.innerText = message;
    statusElement.classList.remove('hidden');
}

function hideStatus() {
    statusElement.classList.add('hidden');
    statusElement.innerText = '';
}

function displayResult(isFake) {
    const container = document.getElementById('result_container');
    const contentInput = document.getElementById('content_input');
    const iconContainer = document.getElementById('result_icon_container');
    const icon = document.getElementById('result_icon');
    const title = document.getElementById('result_title');
    const description = document.getElementById('result_desc');

    container.classList.remove('hidden');
    contentInput.blur();
    contentInput.scrollTop = 0;

    if (isFake) {
        container.classList.remove('bg-slate-50');
        container.classList.add('bg-red-50');
        iconContainer.className = "p-3 rounded-full shrink-0 bg-red-100 text-red-600";
        icon.className = "fas fa-exclamation-triangle text-2xl";
        title.innerText = "Potential fake news detected";
        title.className = "text-xl font-bold mb-1 text-red-900";
        description.innerText = "FaKe found linguistic patterns associated with fake articles in its training data. Verify the article and its claims with reputable sources.";
    } else {
        container.classList.remove('bg-slate-50');
        container.classList.add('bg-green-50');
        iconContainer.className = "p-3 rounded-full shrink-0 bg-green-100 text-green-600";
        icon.className = "fas fa-shield-alt text-2xl";
        title.innerText = "Likely real news";
        title.className = "text-xl font-bold mb-1 text-green-900";
        description.innerText = "FaKe found linguistic patterns associated with real articles in its training data. This result is not a substitute for checking the article's claims and source.";
    }
}

function displayError(message) {
    const container = document.getElementById('result_container');
    const title = document.getElementById('result_title');
    const description = document.getElementById('result_desc');
    const iconContainer = document.getElementById('result_icon_container');
    const icon = document.getElementById('result_icon');

    container.classList.remove('hidden');
    container.classList.add('bg-slate-50');
    iconContainer.className = "p-3 rounded-full shrink-0 bg-slate-200 text-slate-600";
    icon.className = "fas fa-wifi text-2xl";
    title.innerText = "Unable to analyze article";
    title.className = "text-xl font-bold mb-1 text-slate-900";
    description.innerText = message;
}

async function checkIfFakeNews(article) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ news_body: article }),
            signal: controller.signal,
        });

        if (!response.ok) {
            throw new Error(`The prediction service returned HTTP ${response.status}. Please try again.`);
        }

        const responseData = await response.json();
        if (typeof responseData.status !== 'boolean') {
            throw new Error('The prediction service returned an unexpected response.');
        }

        return responseData.status;
    } finally {
        window.clearTimeout(timeout);
    }
}
