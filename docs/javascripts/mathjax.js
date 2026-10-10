window.MathJax = {
  loader: {
    load: ["[tex]/boldsymbol"]
  },
  tex: {
    packages: {"[+]": ["boldsymbol"]},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  // on the first page load MathJax typesets the page itself once its startup has finished
  if (!MathJax.startup || !MathJax.startup.output) return
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
