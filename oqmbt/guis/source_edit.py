"""
Module :mod:`oqmbt.guis.project_select` defines the gui setting the
fundamental parameters of a project used to run scripts.
"""

import os
from ipywidgets import widgets
from oqmbt.guis.project_select import (_get_source_types,
                                       handle_change, handle_change_type,
                                       handle_click)
MARGIN = 5
SKIP_SET = set(['polygon'])
FIXED_SET = set(['source_type'])


def source_editor_gui(project):
    """
    :parameter project:
        An instance of a :class:`oqmbt.project.OQtProject`
    """
    global oqmbtp
    global model
    global w_name, w_dir, w_model, w_sources, project_dir, w_types

    oqmbtp = project
    models = oqmbtp.models.keys()
    project_dir = oqmbtp.directory
    project_pickle_filename = os.environ.get('OQMBT_PROJECT')

    wdg_list = []
    w_title = widgets.HTML(value="<h3>Edit source parameters<h3>")
    # Project name
    tmp_str = "Project:<br>"
    tmp_str += "<small>%s</small><br>" % (project_dir)
    # Filename
    tmp_str += "Filename: <br>"
    tmp_str += "<small>%s</small><br><br>" % (
        os.path.split(project_pickle_filename)[1])
    w_dir = widgets.HTML(value=tmp_str)
    w_name = widgets.Text(description='Name',
                          value=oqmbtp.name,
                          width=600,
                          margin=MARGIN)
    wdg_list.append(w_title)
    wdg_list.append(w_dir)
    wdg_list.append(w_name)

    if len(models):
        model_id = models[0]
        model = oqmbtp.models[model_id]
        w_model = widgets.Dropdown(options=models,
                                   description='Model',
                                   value=model_id)
        if len(model.sources.keys()):
            types = sorted(_get_source_types(model))
            types.insert(0, 'All')
            # Create the list of Text widgets
            param_list = get_source_param_list()
            print(param_list)
            w_text_list, par_dict = param_editor_create(param_list)
            # Source types widget
            w_types = widgets.Select(options=types,
                                     description='Types',
                                     margin=MARGIN,
                                     height=50,
                                     value=types[0])
            keys_list = sorted(model.sources.keys())
            w_sources = widgets.Select(options=keys_list,
                                       description='Sources',
                                       value=keys_list[0],
                                       margin=MARGIN)
            # Set parameters
            param_set(param_list, par_dict)
        else:
            w_sources = widgets.SelectMultiple(options=[],
                                               description='Sources:',
                                               margin=MARGIN)
            w_types = widgets.Select(options=[],
                                     description='Types',
                                     margin=MARGIN)
        wdg_list.append(w_model)
        wdg_list.append(w_types)
        wdg_list.append(w_sources)
        wdg_list = wdg_list + w_text_list

    w_butt = widgets.Button(description='Done!', width=100, border_color='red')
    wdg_list.append(w_butt)

    w_types.on_trait_change(handle_change_type, 'value')
    w_model.on_trait_change(handle_change)
    w_butt.on_click(handle_click)

    return widgets.VBox(children=wdg_list)


def get_source_param_list():
    """
    Get the list of parameters used to descibe the sources in a given model.
    In the future this should be probably added to the project description.
    """
    par_set = set()
    for key in model.sources:
        src = model.sources[key]
        for par in src.__dict__:
            if not SKIP_SET & set([par]):
                par_set = par_set | set([par])
    return list(par_set)


def param_editor_create(par_list):
    """
    This creates a list of text widgets given a list of parameters
    """
    wdg_list = []
    wdg_dict = {}
    wdg_list = [widgets.HTML(value="<h4>Source parameters<h4>")]
    for par in par_list:
        wdg = widgets.Text(description=par,
                           margin=MARGIN,
                           width=500)
        wdg_list.append(wdg)
        wdg_dict[par] = wdg
        del wdg
    return wdg_list, wdg_dict


def param_set(par_list, wdg_dict):
    model_id = w_model.value
    model = oqmbtp.models[model_id]
    src_id = w_sources.value
    src = model.sources[src_id]
    for key in par_list:
        wdg = wdg_dict[key]
        wdg.value = getattr(src, key)
