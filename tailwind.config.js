// tailwind.config.js

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./**/*.html", // Scan all HTML files in the root and subdirectories
    "./src/js/*.js", // Scan all JavaScript files for dynamic classes
  ],
  theme: {
    extend: {
      // You can add custom fonts or colors here
    },
  },
  plugins: [],
}