const API_ENDPOINT = "https://fph-ml.onrender.com/check-news";

document.getElementById('check_button').addEventListener('click', async () => {
    const content = document.getElementById('content_input').value;
    if (!content.trim()) {
        alert('Please enter the content of the news article.');
        return;
    }

    //  Show loading message
    document.getElementById('result').innerText = 'Checking...';

    //  Show result
    try {
        const is_fake = await check_if_fake_news(content);
        document.getElementById('result').innerText = is_fake ? 'This is probably fake!' : 'This is probably real!';
    } catch (error) {
        document.getElementById('result').innerText = `Error: ${error.message}. Please try again.`;
    }
});

//  Consume FaKe API
async function check_if_fake_news(article) {
    const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ news_body: article }),
    });

    if (!response.ok) {
        throw new Error('Network response was not ok!');
    }

    const response_data = await response.json();
    return response_data.status;
}
