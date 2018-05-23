"""
Module :mod:`openquake.mbt.guis.project_select` defines the gui setting the
fundamental parameters of a project used to run scripts.
"""

import os
from ipywidgets import widgets


def project_select_gui(project):
    """
    :parameter project:
        An instance of a :class:`openquake.mbt.project.OQtProject`
    """
    global oqmbtp
    global model
    global w_name, w_dir, w_model, w_sources, project_dir, w_types

    margin = 5

    oqmbtp = project
    models = list(oqmbtp.models.keys())
    project_dir = oqmbtp.directory
    project_pickle_filename = os.environ.get('OQMBT_PROJECT')

    wdg_list = []
    w_title = widgets.HTML(value="<h3>Set model parameters<h3>")
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
                          margin=margin)
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
            w_types = widgets.Select(options=types,
                                     description='Types',
                                     margin=margin,
                                     value=types[0])
            keys_list = sorted(model.sources.keys())
            w_sources = widgets.SelectMultiple(options=sorted(keys_list),
                                               description='Sources',
                                               margin=margin)
        else:
            w_sources = widgets.SelectMultiple(options=[],
                                               description='Sources:')

        wdg_list.append(w_model)
        wdg_list.append(w_types)
        wdg_list.append(w_sources)

    w_butt = widgets.Button(description='Done!', width=100, border_color='red')
    wdg_list.append(w_butt)

    w_types.on_trait_change(handle_change_type, 'value')
    w_model.on_trait_change(handle_change)
    w_butt.on_click(handle_click)

    return widgets.VBox(children=wdg_list)


def _get_source_types(model):
    """
    :parameter dict sources:
        A dictionary of :class:`openquake.mbt.oqt_project.OQtSource` instances
    :returns:
        A list of source types
    """
    out_set = set()
    for key in model.sources:
        out_set.add(model.sources[key].source_type)
    out_list = list(out_set)
    return out_list


def _get_keys(source_types):
    """
    :parameter dict sources:
        A dictionary of :class:`openquake.mbt.oqt_project.OQtSource` instances
    :parameter list source_types:
        A list of source typologies to be extracted
    :returns:
        A list of source IDs
    """
    types_set = set([source_types])
    if types_set & set(['All']):
        out_list = []
        for key in model.sources:
            if set([model.sources[key].source_type]) | types_set:
                out_list.append(key)
    else:
        out_list = []
        for key in model.sources:
            if set([model.sources[key].source_type]) & types_set:
                out_list.append(key)
    return out_list


def handle_change(sender):
    """
    This handles cases when the user changes the text of the widget with the
    name of the project.

    :parameter sender:
        This is
    """
    model_id = w_model.value
    model = oqmbtp.models[model_id]
    if len(model.sources.keys()):
        w_sources.options = model.sources.keys()
    else:
        w_sources.options = []


def handle_click(sender):
    """
    This handles cases when the user changes the text of the widget with the
    name of the project.

    :parameter sender:
        This is
    """
    oqmbtp.name = w_name.value
    oqmbtp.active_model_id = w_model.value
    oqmbtp.active_source_id = w_sources.value
    oqmbtp.save()


def handle_change_type(sender, value):
    """
    """
    source_types = w_types.value
    src_key_list = _get_keys(source_types)
    w_sources.options = sorted(src_key_list)
