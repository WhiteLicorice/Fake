// ==UserScript==
// @name         FaKe
// @namespace    https://github.com/WhiteLicorice
// @version      2.6.0
// @description  Fake news classification script that interfaces with an ML model. Uses heuristic-driven scraping with noise filtering and iframe protection.
// @author       Rene Andre Jocsing, Kobe Austin Lupac, Chancy Ponce de Leon, Ron Gerlan Naragdao
// @icon         https://cdn0.iconfinder.com/data/icons/modern-fake-news/500/asp1430a_9_newspaper_fake_news_icon_outline_vector_thin-1024.png
// @grant        GM_registerMenuCommand
// @grant        GM_addStyle
// @match        *://*/*
// @connect      fph-ml.onrender.com
// @connect      fake-ph.cyclic.cloud
// @connect      localhost
// ==/UserScript==

(function () {
	'use strict';

	// 1. ANTI-IFRAME GUARD (CRITICAL)
	// Ensures script only runs on the main page, not on ads/embeds.
	if (window.self !== window.top) { return; }

	const API_ENDPOINT = "https://fph-ml.onrender.com/check-news";

	// --- STYLES ---
	GM_addStyle(`
        .fake-ext-toast {
            position: fixed; bottom: 20px; right: 20px;
            background: #ffffff; color: #333;
            padding: 16px 24px; border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            z-index: 99999; display: flex; align-items: center; gap: 15px;
            transform: translateY(100px); opacity: 0;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border-left: 6px solid #ccc; max-width: 350px;
        }
        .fake-ext-toast.visible { transform: translateY(0); opacity: 1; }
        .fake-ext-spinner {
            width: 20px; height: 20px; border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db; border-radius: 50%;
            animation: fake-spin 1s linear infinite;
        }
        @keyframes fake-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .fake-ext-result-icon { font-size: 24px; }
        .fake-ext-content strong { display: block; font-size: 14px; margin-bottom: 2px; }
        .fake-ext-content span { font-size: 12px; color: #666; }
    `);

	// --- MENU COMMAND ---
	GM_registerMenuCommand("Check Current Article", run_script_pipeline);

	// --- UI HELPER FUNCTIONS ---
	let toastElement = null;
	function getToast() {
		if (!toastElement) {
			toastElement = document.createElement('div');
			toastElement.className = 'fake-ext-toast';
			document.body.appendChild(toastElement);
		}
		return toastElement;
	}

	function showStatus(type, title, message) {
		const toast = getToast();
		let iconHtml = '', borderColor = '#ccc';

		if (type === 'loading') { iconHtml = '<div class="fake-ext-spinner"></div>'; borderColor = '#3498db'; }
		else if (type === 'real') { iconHtml = '<div class="fake-ext-result-icon">✅</div>'; borderColor = '#2ecc71'; }
		else if (type === 'fake') { iconHtml = '<div class="fake-ext-result-icon">⚠️</div>'; borderColor = '#e74c3c'; }
		else if (type === 'error') { iconHtml = '<div class="fake-ext-result-icon">❌</div>'; borderColor = '#95a5a6'; }

		toast.style.borderLeftColor = borderColor;
		toast.innerHTML = `${iconHtml}<div class="fake-ext-content"><strong>${title}</strong><span>${message}</span></div>`;
		requestAnimationFrame(() => toast.classList.add('visible'));
		if (type !== 'loading') setTimeout(() => toast.classList.remove('visible'), 8000);
	}

	// --- MAIN LOGIC (REVERTED TO FLAT LIST) ---

	async function scrape_paragraphs() {
		// Simple, robust strategy:
		// 1. Select all Paragraph (<p>) elements. 
		// 2. Filter out anything too short (likely menu items, bylines, or captions).
		// 3. Join them with newlines.

		const paragraphs = Array.from(document.querySelectorAll('p'));

		const cleanText = paragraphs
			.map(p => p.innerText.trim())
			.filter(text => text.length > 50) // Heuristic: Sentences usually have > 50 chars. Ugh.
			.join("\n\n");

		return cleanText;
	}

	async function is_fake_news(articleText) {
		try {
			const payload = { news_body: articleText };
			const response = await fetch(API_ENDPOINT, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			});
			if (!response.ok) throw new Error(`Server error: ${response.status}`);
			return await response.json();
		} catch (error) {
			console.error("FaKe API Error:", error);
			throw error;
		}
	}

	async function run_script_pipeline() {
		showStatus('loading', 'Analyzing Article', 'FaKe is scanning page content...');

		try {
			const processed_article = await scrape_paragraphs();

			if (!processed_article || processed_article.length < 100) {
				showStatus('error', 'No text found', 'Could not find enough paragraph text to analyze.');
				return;
			}

			const result = await is_fake_news(processed_article);

			// Backend Logic: status=True means FAKE
			const isFake = result.status === true;

			if (isFake) {
				showStatus('fake', 'Potential Fake News', 'FaKe flagged this content.');
			} else {
				showStatus('real', 'Likely Credible', 'FaKwe found no issues.');
			}

		} catch (e) {
			showStatus('error', 'Connection Failed', 'Could not reach the analysis server.');
		}
	}
})();