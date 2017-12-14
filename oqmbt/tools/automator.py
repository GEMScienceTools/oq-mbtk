
import os
import subprocess

from oqmbt.tools import notebook
from oqmbt.oqt_project import OQtProject


def _get_report_path_name(project_dir, model_id, src_str, notebook_name):
    """
    Returns the path and the name of the report
    :parameter project_filename:
        The name of the file (it's a pickle file) containing the project
    :parameter model_id:
        A string identifying the ID of the model with the sources to be
        processed
    :parameter src_str:
        A string identifying uniquely the source (generally it's the source ID)
    :parameter notebook_name:
        The name of the notebook used for the processing
    """
    nb_name = os.path.splitext(notebook_name)[0]
    # src_str = '_'.join(src_list)
    rpt_name = 'rpt-%s-%s-%s.html' % (nb_name, model_id, src_str)
    rpt_path = os.path.join(project_dir, 'reports')
    rpt_path = os.path.join(rpt_path, model_id)
    return rpt_path, rpt_name


def run(project_filename, model_id, notebook_path, src_id_list,
        reports_folder=''):
    """
    :parameter project_filename:
        The name of the file (it's a pickle file) containing the project
    :parameter model_id:
        A string identifying the ID of the model with the sources to be
        processed
    :parameter notebook_path:
        The path to the notebook to be executed
    :parameter src_id_list:
        A list containing the list of the sources to be processed
    :parameter reports_folder:
        The name of the folder where to create the .html report. An empty
        string does not trigger the creation of a report.
    :returns:
        A tuple containing an instance of the class
        `nbformat.notebooknode.NotebookNode` and a dictionary
    """
    #
    # check that the output folder exists
    if len(reports_folder) and not os.path.exists(reports_folder):
        os.makedirs(reports_folder)
    #
    # options
    opt = ''
    #
    # running
    for idx, elem in enumerate(sorted(src_id_list)):
        #
        # saving project
        oqmbtp = OQtProject.load_from_file(project_filename)
        oqmbtp.active_model_id = model_id
        oqmbtp.active_source_id = [elem]
        oqmbtp.save()
        #
        # output directory
        outdir = oqmbtp.directory
        del oqmbtp
        #
        #
        rpt_name = None
        if len(reports_folder):
            nb_name = os.path.split(notebook_path)[1]
            rpt_path, rpt_name = _get_report_path_name(outdir,
                                                       model_id,
                                                       elem,
                                                       nb_name)
        #
        #
        msg = 'Running {:s}'.format(os.path.basename(notebook_path))
        msg += ' for source with ID {:s}'.format(elem)
        print (msg)
        #
        # running the notebook
        out = notebook.run(notebook_path, '', reports_folder=reports_folder,
                           key=rpt_name)
        #
        #
        """
        if len(reports_folder):
            nb_name = os.path.split(notebook_path)[1]
            rpt_path, rpt_name = _get_report_path_name(outdir,
                                                       model_id,
                                                       elem,
                                                       nb_name)

            path = os.path.dirname(notebook_path)
            html_filename = os.path.join(rpt_path, rpt_name)
            in_html = os.path.join(rpt_path, os.path.splitext(nb_name)[0])

            print (in_html, html_filename)

            cmd_str = 'mv %s.html %s' % (in_html,
                                         html_filename)
            out = subprocess.call(cmd_str, shell=True)
            if out:
                print ('Error in moving the report to the final folder')
                print ('from: %s.html' % (os.path.splitext(nb_name)[0]))
                print ('to:', html_filename)
                return 1
            else:
                print ('Created %s' % (rpt_name))
        """
