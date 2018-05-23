"""
"""

from IPython.display import display
from IPython.core.display import HTML

txt_ok = """
  <div class="alert alert-success fade in">
    <a href="#" class="close" data-dismiss="alert"
        aria-label="close">&times;</a>
    <strong>Success!</strong> execution completed successfully.
  </div>
"""

txt_ko = """
  <div class="alert alert-danger fade in">
    <a href="#" class="close" data-dismiss="alert"
        aria-label="close">&times;</a>
    <strong>Error!</strong> execution not completed successfully.
  </div>
"""


def processing_out(out):

    if out is True:
        display(HTML(txt_ok))
    else:
        display(HTML(txt_ko))
