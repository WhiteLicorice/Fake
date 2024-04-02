// Display the loader when the page loads
document.addEventListener("DOMContentLoaded", function() {
    document.getElementById('loader').style.display = 'block';
});

// Simulating API call delay
setTimeout(() => {
    const api_result = Math.random() < 0.5;

    document.getElementById('loader').style.display = 'none';

    const isFakeNews = api_result === true;
    const message = isFakeNews ? "Fake_API says this is probably FAKE!!!" : "Fake_API says this is probably REAL!!!";
    alert(message);
}, 3000); // Simulating 3-second API response delay
