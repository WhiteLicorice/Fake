// ==UserScript==
// @name         Fake News Detector
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  A userscript that connects with a Node-hosted machine learning model to determine if an article or a post is authentic or otherwise.
// @author       Rene Andre Jocsing, Kobe Austin Lupac, Chancy Ponce de Leon
// @icon         https://cdn0.iconfinder.com/data/icons/modern-fake-news/500/asp1430a_9_newspaper_fake_news_icon_outline_vector_thin-1024.png
// @grant        none
// @match https://www.philstar.com/*
// ==/UserScript==

(function() {
    'use strict';

    console.log("The script is live!")

    var really_fake = false//   Scaffold for machine learning model return value

    function is_fake_news(news_article) {
        //SCAFFOLD
        //TODO: Integrate machine learning model in the backend

        return really_fake

    }

    function scrape_paragraphs(){
        var paragraphs = document.querySelectorAll("p")//  Returns an array of all the paragraph elements on the page
        var news_article = []//  Initialize an array to contain all the paragraph.textContents
        try {
            paragraphs.forEach(paragraph => {
                const paragraph_text = paragraph.textContent.trim()
                if (paragraph_text.length > 0) {
                    news_article.push(paragraph_text)
                }
            })
        } catch(error) {
            console.log('Error parsing paragraphs occured: ${error}')
        } finally {
            console.log(news_article.join("\n"))
            var fake_news = is_fake_news(news_article.join("\n"))
            if (fake_news) {
                alert("This is FAKE!!!")
            }
            else {
                alert("This is REAL!!!")
            }
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
        floatingButton.addEventListener("click", scrape_paragraphs);
    
        // Append the button to the body of the page
        document.body.appendChild(floatingButton);
    }

    add_floating_button()

})();