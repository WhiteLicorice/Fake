// ==UserScript==
// @name         Fake News Detector
// @namespace    http://tampermonkey.net/
// @version      1.0.1
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

	//console.log("The script is live!")

	//var API_ENDPOINT = "http://127.0.0.1:5000/check-news" // Localhost endpoint
   	//var API_ENDPOINT = "https://fake-ph.cyclic.cloud/check-news" // Cyclic.sh endpoint for old SVC model
   	var API_ENDPOINT = "https://fph-ml.onrender.com/check-news" // Render endpoint for new LogisticRegression model

	async function run_script_pipeline(){
		var processed_article = await scrape_paragraphs()
		var fake_api_result = await is_fake_news(processed_article)
		display_is_fake_news(fake_api_result)
	}

	async function display_is_fake_news(api_result){
	        console.log(api_result)
	        const isFakeNews = api_result === true;
	        const message = isFakeNews ? "Fake_API says this is probably FAKE!!!" : "Fake_API says this is probably REAL!!!";
	        alert(message);
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
            console.log(error.message)
            alert("Error connecting to Fake_API! Please try again!")
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
		var paragraphs = document.querySelectorAll("p")//  Returns an array of all the paragraph elements on the page
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
