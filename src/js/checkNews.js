const API_ENDPOINT = "https://fph-ml.onrender.com/check-news";

document.getElementById('check_button').addEventListener('click', async () => {
    const content = document.getElementById('content_input').value;
    const btn = document.getElementById('check_button');
    const resultContainer = document.getElementById('result_container');

    // Reset UI
    resultContainer.classList.add('hidden');
    resultContainer.className = "hidden bg-slate-50 border-t border-slate-100 p-6 md:p-8 animate-fade-in"; // Reset colors

    if (!content.trim() || content.length < 20) {
        alert('Please enter a substantial amount of text (at least 20 characters) to analyze.');
        return;
    }

    // Set Loading State
    const originalBtnText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Analyzing...';
    btn.disabled = true;

    try {
        const is_fake = await check_if_fake_news(content);
        displayResult(is_fake);
    } catch (error) {
        displayError(error.message);
    } finally {
        // Reset Button
        btn.innerHTML = originalBtnText;
        btn.disabled = false;
    }
});

function displayResult(isFake) {
    const container = document.getElementById('result_container');
    const iconContainer = document.getElementById('result_icon_container');
    const icon = document.getElementById('result_icon');
    const title = document.getElementById('result_title');
    const desc = document.getElementById('result_desc');

    container.classList.remove('hidden');

    if (isFake) {
        // FAKE NEWS UI
        container.classList.remove('bg-slate-50');
        container.classList.add('bg-red-50');

        iconContainer.className = "p-3 rounded-full shrink-0 bg-red-100 text-red-600";
        icon.className = "fas fa-exclamation-triangle text-2xl";

        title.innerText = "Potential Fake News Detected";
        title.className = "text-xl font-bold mb-1 text-red-900";
        desc.innerText = "FaKe has flagged this content as likely misleading or fabricated. Please verify with other reputable sources.";
    } else {
        // REAL NEWS UI
        container.classList.remove('bg-slate-50');
        container.classList.add('bg-green-50');

        iconContainer.className = "p-3 rounded-full shrink-0 bg-green-100 text-green-600";
        icon.className = "fas fa-shield-alt text-2xl";

        title.innerText = "Likely Credible";
        title.className = "text-xl font-bold mb-1 text-green-900";
        desc.innerText = "FaKe did not find significant indicators of fake news in this text.";
    }
}

function displayError(msg) {
    const container = document.getElementById('result_container');
    const title = document.getElementById('result_title');
    const desc = document.getElementById('result_desc');
    const iconContainer = document.getElementById('result_icon_container');
    const icon = document.getElementById('result_icon');

    container.classList.remove('hidden');
    container.classList.add('bg-slate-50');

    iconContainer.className = "p-3 rounded-full shrink-0 bg-slate-200 text-slate-600";
    icon.className = "fas fa-wifi text-2xl";

    title.innerText = "Connection Error";
    title.className = "text-xl font-bold mb-1 text-slate-900";
    desc.innerText = `We couldn't reach the server. ${msg}`;
}

// Consume FaKe API
async function check_if_fake_news(article) {
    const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ news_body: article }),
    });

    if (!response.ok) {
        throw new Error('Network response was not ok');
    }

    const response_data = await response.json();

    // Handle specific backend response structure
        return response_data.status;
}