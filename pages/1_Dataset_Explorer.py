"""Dataset Explorer module."""

from utils import get_module, render_module_placeholder, render_page_sidebar

_MODULE = get_module("Dataset Explorer")

render_page_sidebar(_MODULE)
render_module_placeholder(_MODULE)
