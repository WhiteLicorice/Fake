let lastScrollTop = 0;
const navbar = document.getElementById('navbar');
let navbarHeight = navbar.offsetHeight;

window.addEventListener('scroll', function() {
  let scrollTop = window.pageY || document.documentElement.scrollTop;

  if (scrollTop > lastScrollTop) {
    // Scroll down
    navbar.style.top = `-${navbarHeight}px`;
  } else {
    // Scroll up
    navbar.style.top = '0';
  }

  lastScrollTop = scrollTop;
});