// ==UserScript==
// @name         Fake News Detector
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  A userscript that connects with a Node-hosted machine learning model to determine if an article or a post is authentic or otherwise.
// @author       Rene Andre Jocsing, Kobe Austin Lupac, Chancy Ponce de Leon
// @match        https://twitter.com/*
// @match        https://www.facebook.com/*
// @icon         https://cdn0.iconfinder.com/data/icons/modern-fake-news/500/asp1430a_9_newspaper_fake_news_icon_outline_vector_thin-1024.png
// @grant        none
// ==/UserScript==




(function main() {
    'use strict';

    console.log("The script is live!")

    //    TODO: Ping a backend server hosted on some port and spoof results.

    function isTweet(element) {
        //  TODO: Return True if the element attribute data-testid = "tweet" else False
    }
    
    function addButtonTweeter(element) {
        //  TODO: Add a "Validate" button to all nodes that are tweets
        //  HINT: Use GM_addElement(parent_node, tag_name, attributes) method, where parent_node is the node to attach the button, tag_name is the HTML tag of the node to be attached, and attributes is a dictionary of additional attributes
        //  EXAMPLE:
        /* 
        GM_addElement('script', {
            textContent: 'window.foo = "bar";'
        });

        GM_addElement('script', {
            src: 'https://example.com/script.js',
            type: 'text/javascript'
        });

        GM_addElement(document.getElementsByTagName('div')[0], 'img', {
            src: 'https://example.com/image.png'
        });

        GM_addElement(shadowDOM, 'style', {
            textContent: 'div { color: black; };'
        }); 
        */
    }


})();