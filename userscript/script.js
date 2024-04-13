// ==UserScript==
// @name         Fake News Detector
// @namespace    http://tampermonkey.net/
// @version      1.0.2
// @description  A userscript that interfaces with a cloud-hosted machine learning model to determine if an article is fake news.
// @author       Rene Andre Jocsing, Kobe Austin Lupac, Chancy Ponce de Leon, Ron Gerlan Naragdao
// @icon         https://cdn0.iconfinder.com/data/icons/modern-fake-news/500/asp1430a_9_newspaper_fake_news_icon_outline_vector_thin-1024.png
// @grant        GM_registerMenuCommand
// @grant        GM_addStyle
// @match *://*/*
// @connect https://fake-ph.cyclic.cloud
// @connect https://fph-ml.onrender.com/check-news
// @connect localhost
// ==/UserScript==

(function() {
	'use strict';

    const spinnerHTML = `
    <div id="spinner-container" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 9999;">
        <section class="spinner-dots">
            <span style="--dot: 1;"></span>
            <span style="--dot: 2;"></span>
            <span style="--dot: 3;"></span>
            <span style="--dot: 4;"></span>
            <span style="--dot: 5;"></span>
            <span style="--dot: 6;"></span>
            <span style="--dot: 7;"></span>
            <span style="--dot: 8;"></span>
            <span style="--dot: 9;"></span>
        </section>
    </div>
    `;

        // Insert the spinner HTML into the document body
    document.body.insertAdjacentHTML('beforeend', spinnerHTML);

    const spinnerCss = `
    <style>
        section.spinner-dots {
            display: flex;
            height: 100vh;
            width: 100%;
            align-items: center;
            justify-content: center;
        }

        section.spinner-dots span {
            position: absolute;
            height: calc(10px + var(--dot) * 1px);
            width: calc(10px + var(--dot) * 1px);
            background: white;
            border-radius: 50%;
            transform: rotate(calc(var(--dot) * (360deg / 9))) translateY(35px);
            animation: animate 1.5s linear infinite;
            animation-delay: calc(var(--dot) * 0.1s);
            opacity: 0;
        }

        @keyframes animate {
            0% {
                opacity: 1;
            }
            100% {
                opacity: 0;
            }
        }
    </style>
    `;

    // Insert the spinner CSS into the head of the document
    document.head.insertAdjacentHTML('beforeend', spinnerCss);


	// blurs the whole page while loading
	const overlayDiv = document.createElement('div');
    overlayDiv.id = 'overlay';
    overlayDiv.style.cssText = 'display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5); backdrop-filter: blur(5px); z-index: 9998;';

    document.body.appendChild(overlayDiv);

    function showSpinner(){
        document.getElementById('spinner-container').style.display = 'block';
    }
    function hideSpinner(){
        document.getElementById('spinner-container').style.display = 'none';
    }

    function showOverlay(){
        document.getElementById('overlay').style.display = 'block';
    }
    function hideOverlay(){
        document.getElementById('overlay').style.display = 'none';
    }

	//console.log("The script is live!")

	//	TODO: Handle the case where the connection fails by removing the spinner and overlay
	//	TODO: Place all files for extension /userscript

	//var API_ENDPOINT = "http://127.0.0.1:5000/check-news" // Localhost endpoint
   	//var API_ENDPOINT = "https://fake-ph.cyclic.cloud/check-news" // Depreciated Cyclic endpoint
   	var API_ENDPOINT = "https://fph-ml.onrender.com/check-news" // Render endpoint

       async function run_script_pipeline(){
        // Show loading spinner
        showSpinner();
		// show overlay
		showOverlay();

        var processed_article = await scrape_paragraphs();
        if (processed_article.trim().length === 0) {
            display_unable_to_scrape();
            // Hide loading spinner
            hideSpinner();
			hideOverlay();
            return;
        }
        var fake_api_result = await is_fake_news(processed_article);
        display_is_fake_news(fake_api_result);
        // Hide loading spinner
        hideSpinner();
		hideOverlay();
    }


	async function display_is_fake_news(api_result){
		// Show loading spinner
		showSpinner();
		// show overlay
		showOverlay();

		setTimeout(() => {
			hideSpinner();
			hideOverlay();

			// Create a custom modal
			const customAlert = `
				<div id="custom-modal" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.3); z-index: 9999;">
					<h2 style="margin-bottom: 20px;">Fake_API Result</h2>
					<p style="font-weight: bold; text-align: center;">Fake_API says this is probably ${api_result ? 'FAKE' : 'REAL'}!!!</p>
					<button id="close-modal-btn" style="display: block; margin: 20px auto; padding: 10px 20px; background-color: #007bff; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Close</button>
				</div>
			`;

			// Insert the modal HTML into the body
			document.body.insertAdjacentHTML('beforeend', customAlert);

			// Close modal button functionality
			const closeModalBtn = document.getElementById('close-modal-btn');
			closeModalBtn.addEventListener('click', () => {
				const customModal = document.getElementById('custom-modal');
				customModal.parentNode.removeChild(customModal);
			});
		}, 100);
	}


	async function display_unable_to_scrape(){
		const reportLink = 'https://github.com/WhiteLicorice/Fake/issues/new';
		const alertMessage = "FaKe extension was unable to scrape content.\nPlease try again.\nIf the issue persists, please report the website on GitHub."
		const userChoice = confirm(`${alertMessage}\nClick 'OK' to report the issue on GitHub.\nThis will open a pop-up page.`);
		if (userChoice) {
			window.open(reportLink, '_blank');
		}
	}

	async function is_fake_news(news_article) {
        try {
            var FAKE_API_CALL = await fetch(API_ENDPOINT, {
			                    method: 'POST',
			                    body: JSON.stringify({ news_body: news_article }), // Assuming news_article is a string
			                    headers: {
				                       'Content-Type': 'application/json', // Set the content type to JSON
			                    },
		    });

        } catch (error) {
			setTimeout(() => {
			hideSpinner();
			hideOverlay();
				console.log(error.message)
				alert("Error connecting to Fake_API! Please try again!")
			}, 100); // Adjust the delay as needed

}

		if (!FAKE_API_CALL.ok) {
			throw new Error('Network response was not ok');
		}

		try {
			const responseData = await FAKE_API_CALL.json(); // Parse the response as JSON
            //console.log(responseData)
			return responseData.status
		} catch (error) {
			console.error('API Request Error:', error);
		}
	}

	async function scrape_paragraphs(){
		var paragraphs = document.querySelectorAll("p, span")//  Returns an array of all the paragraph and span elements on the page
        console.log(paragraphs)
		var news_article = []//  Initialize an array to contain all the paragraph.textContents
		try {
			paragraphs.forEach(paragraph => {
				var paragraph_text = paragraph.textContent.trim()
				if (paragraph_text.length > 0) {
					news_article.push(paragraph_text)
				}
			})
			console.log(news_article.join("\n"))
			return (news_article.join("\n"))
		} catch(error) {
			console.log('Error parsing paragraphs occured: ${error.message}')
		}
	}

	function add_floating_button(){
		// Create a floating button element
		const floatingButton = document.createElement("button")
		floatingButton.textContent = "DETECT FAKE NEWS"
		floatingButton.style.position = "fixed"
		floatingButton.style.bottom = "20px"
		floatingButton.style.right = "20px"
		floatingButton.style.zIndex = "9999"

		// Attach a click event listener to the button
		floatingButton.addEventListener("click", run_script_pipeline);

		// Append the button to the body of the page
		document.body.appendChild(floatingButton);
	}

    const detect_fake_news_command = GM_registerMenuCommand("Detect Fake News", function(MouseEvent) {// Add menu entry to Tampermonkey for cleaner UI
		run_script_pipeline()
	}, {
		accessKey: "f",
		autoClose: true
	});

	//add_floating_button()

})();
