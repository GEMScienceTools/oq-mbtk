"""
Module :mod:`openquake.mbt.guis.automator` defines the gui controlling
the execution of other notebooks
"""

import os
import subprocess

from glob import glob

from ipywidgets import widgets

from IPython.core.display import HTML
from IPython.display import display

from openquake.mbt.oqt_project import OQtProject

NB_TYPES = sorted(['catalogue',
                   'project',
                   'sources_area',
                   'sources_shallow_fault',
                   'tectonics'])


def automator_gui(filename):
    """
    :parameter project_filename:
        The name of a puckle file containing an OQtProject
    """
    global w_model, w_sources, w_nb_name, w_nb_type, w_repo, w_progress
    global project_filename, model
    global project_dir
    wdg_list = []

    margin = 5

    project_filename = filename
    oqmbtp = OQtProject.load_from_file(project_filename)
    models = oqmbtp.models.keys()
    project_dir = oqmbtp.directory

    w_title = widgets.HTML(value="<h3>Automator<h3>")
    tmp_str = "Name     : %s <br>" % (oqmbtp.name)
    tmp_str += "Stored in: %s <br><br>" % (project_dir)
    w_text = widgets.HTML(value=tmp_str)
    wdg_list.append(w_title)
    wdg_list.append(w_text)

    tmp_str = "Warning: the model does not contain sources"
    w_warn = widgets.HTML(value=tmp_str, visible=False)

    if len(models):
        model_id = models[0]
        model = oqmbtp.models[model_id]
        w_model = widgets.Dropdown(options=models,
                                   description='Model',
                                   value=model_id,
                                   width=400,
                                   margin=margin)
        if len(model.sources.keys()):

            # Sources drop down menu
            tmp_list = sorted(model.sources.keys())
            tmp_list.insert(0, 'All')
            tmp_str = 'Sources'
            w_sources = widgets.SelectMultiple(options=tmp_list,
                                               description=tmp_str,
                                               width=200,
                                               margin=margin)
        else:
            w_sources = widgets.Dropdown(options=[],
                                         description='Source')
        wdg_list.append(w_model)
        wdg_list.append(w_sources)
    else:
        w_warn.visible = True

    # Notebook type
    w_nb_type = widgets.Dropdown(options=NB_TYPES,
                                 description='Notebook type',
                                 width=400,
                                 margin=margin)
    wdg_list.append(w_nb_type)

    # Notebook name
    w_nb_name = widgets.Dropdown(options=[],
                                 description='Notebook name',
                                 width=400,
                                 margin=margin)
    wdg_list.append(w_nb_name)

    # Report checkbox
    w_repo = widgets.Checkbox(description='Generate report', value=False)
    wdg_list.append(w_repo)

    # Warning
    wdg_list.append(w_warn)

    # Button
    w_butt = widgets.Button(description='Run', width=100, border_color='red')
    wdg_list.append(w_butt)

    # Progress bar
    w_progress = widgets.FloatProgress(value=0.0,
                                       min=0.0,
                                       step=1,
                                       visible=False,
                                       description='Processing:')
    wdg_list.append(w_progress)

    w_model.on_trait_change(handle_change_model)
    w_nb_type.on_trait_change(handle_change_nb_type, 'value')
    w_butt.on_click(handle_run)

    # Clean variables
    del oqmbtp

    return widgets.VBox(children=wdg_list)


def handle_run(sender):

    # Case all
    if w_sources.value[0] == 'All':
        tlist = sorted(model.sources.keys())
    else:
        tlist = w_sources.value

    # Update progress bar
    w_progress.max = len(tlist)-1
    w_progress.visible = True
    w_progress.value = 0

    # Option
    opt = ' -o' if w_repo.value else ''

    if len(opt) > 0:
        rpt_table = ReportTable()

    # Running
    for idx, elem in enumerate(tlist):

        # Saving project
        oqmbtp = OQtProject.load_from_file(project_filename)
        oqmbtp.active_model_id = w_model.value
        oqmbtp.active_source_id = [elem]
        oqmbtp.save()
        del oqmbtp

        # Running notebook
        nb_str = os.path.join('./%s/%s' % (w_nb_type.value, w_nb_name.value))
        cmd_str = 'runipy %s %s' % (opt, nb_str)
        if len(opt) > 0:
            rpt_path, rpt_name = _get_report_path_name(project_dir,
                                                       w_model.value,
                                                       elem,
                                                       w_nb_name.value)
            if not os.path.exists(rpt_path):
                os.makedirs(rpt_path)

        # Running notebook
        subprocess.call(cmd_str, shell=True)

        if len(opt) > 0:

            # Converting notebook
            cmd_str = 'ipython nbconvert --to html %s' % (nb_str)
            out = subprocess.call(cmd_str, shell=True)
            if out:
                print('Error in creating the html report')

            # Moving report to folder and rename it
            cmd_str = 'mv %s.html %s' % (os.path.splitext(w_nb_name.value)[0],
                                         os.path.join(rpt_path, rpt_name))
            out = subprocess.call(cmd_str, shell=True)
            if out:
                print('Error in moving the report to the final folder')

            # Update reports table
            path = os.path.relpath(os.path.join(rpt_path, rpt_name))
            rpt_table.append_item(path, rpt_name)

        w_progress.value = idx

    if len(opt) > 0:
        display(HTML(rpt_table.get_report_table()))


def _get_report_path_name(project_dir, model_id, src_str, notebook_name):
    """
    Returns the path and the name of the report
    """
    nb_name = os.path.splitext(notebook_name)[0]
    # src_str = '_'.join(src_list)
    rpt_name = 'rpt-%s-%s-%s.html' % (nb_name, model_id, src_str)
    rpt_path = os.path.join(project_dir, 'reports')
    rpt_path = os.path.join(rpt_path, model_id)
    return rpt_path, rpt_name


def handle_change_model(sender):
    """
    """
    global model

    oqmbtp = OQtProject.load_from_file(project_filename)
    model_id = w_model.value
    model = oqmbtp.models[model_id]
    del oqmbtp

    if len(model.sources.keys()):
        w_sources.options = model.sources.keys()
    else:
        w_sources.options = []


def handle_change_nb_type(sender):
    """
    """
    lst = []
    files = glob('./%s/*.ipynb' % w_nb_type.value)
    for key in files:
        lst.append(os.path.split(key)[1])
    if not len(files):
        lst = ['']
    w_nb_name.options = lst


class ReportTable():

    def __init__(self):
        text = '<table style="width:100%">\n'
        text += '<tr><td><b>Link to report</b></td></tr>\n'
        self._text = text

    def append_item(self, report_path, report_filename):
        """
        Add an item to the table and create the html in the corresponding
        folder containing the reports
        """
        link = '<a href="%s" target="_blank">%s</a>' % (report_path,
                                                        report_filename)
        self._text += '<tr><td>%s</td></tr>\n' % (link)

    def get_report_table(self):
        self._text += "</table>"
        return self._text
