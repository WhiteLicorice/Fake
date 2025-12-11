const navbar = document.getElementById('navbar');
let lastScrollTop = 0;

window.addEventListener('scroll', function () {
  let scrollTop = window.pageY || document.documentElement.scrollTop;

  // Add shadow when scrolled
  if (scrollTop > 10) {
    navbar.classList.add('shadow-md');
  } else {
    navbar.classList.remove('shadow-md');
  }

  lastScrollTop = scrollTop;
});