#!/usr/bin/env python
# coding: utf-8

import re

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3, A2

from reportlab.lib.colors import red, black, blue, grey
from openquake.baselib import sap
from openquake.hazardlib.nrml import to_python


def get_txt(text):
    tmp = re.split('\\n', text)
    otext = [f'{tmp[0]}']
    for t in tmp[1:]:
        otext.append(re.sub('\t','', t))
    return otext


def main(fname, *, outname='report_lt.pdf'):

    # Settings
    landscape = False
    dfontsize = -4
    delta_dx = -2

    lt = to_python(fname)

    # This controls the vertical separation of text describing the value of
    # each branch
    row = 15
    # This is width of each row
    col = 180

    start_x = 30
    start_y = 30
    dx = 5 - delta_dx

    # This controls the height where the description of branches starts
    ny = 8

    if landscape:
        c = canvas.Canvas(outname, pagesize=(A2[1], A2[0]))
        topy = A2[0]
    else:
        c = canvas.Canvas(outname, pagesize=(A2[0], A2[1]))
        topy = A2[1]

    # Header
    c.setFont("Helvetica-Bold", 14+dfontsize)
    tmp = f"Logic Tree ID: {lt.attrib['logicTreeID']}"
    c.drawString(start_x, topy-start_y, tmp)
    c.setFont("Helvetica", 8+dfontsize)
    c.setFillColor(grey)
    c.drawString(start_x, topy-start_y-13, f"File: {fname}")

    # Looping over BranchSets
    i = 0
    tickx = 10
    tickix = 2
    for n, node in enumerate(lt.nodes):

        colx = i*col+start_x

        c.setFont("Helvetica", 12+dfontsize)
        c.setFillColor(red)
        c.drawString(colx, topy-(3*row+start_y),
                     "[{:d}] BSet".format(n))
        c.setFillColor(blue)
        c.drawString(colx+50, topy-(3*row+start_y),
                     "{:s}".format(node.attrib['branchSetID']))
        c.setFillColor(black)
        c.setFont("Helvetica", 10+dfontsize)
        c.drawString(colx, topy-(4*row+start_y),
                     "u: {:s}".format(node.attrib['uncertaintyType']))

        cnt = 1
        for key in node.attrib:
            if re.search('^apply', key):
                c.drawString(colx, topy-((4+cnt)*row+start_y),
                             "a: {:s}".format(key))
                c.drawString(colx, topy-((4+cnt+1)*row+start_y),
                             "   {:s}".format(node.attrib[key]))
                cnt += 2
        c.setFont("Helvetica", 12+dfontsize)

        # Looping over Branches
        for j, snode in enumerate(node.nodes):

            # Branch
            c.setFillColor(red)
            cy = topy-((ny+j*5)*row+start_y)

            # Branch ID
            c.drawString(colx, cy, "B:")
            c.setFillColor(blue)
            c.drawString(i*col+start_x+10, topy-((ny+j*5)*row+start_y),
                         f"{snode.attrib['branchID']}")

            # Branch line
            c.setFillColor(black)
            p = c.beginPath()
            p.moveTo(colx-tickix, cy)
            p.lineTo(colx-tickx, cy)
            c.drawPath(p, fill=1)

            # Branch values
            c.setFont("Helvetica", 10+dfontsize)
            txts = get_txt(snode.nodes[0].text)
            for k, t in enumerate(txts):
                tdy = 0.5+0.5*k
                ttxt = t if k == 0 else f"v: {t}"
                c.drawString(i*col+start_x+dx, topy-((ny+j*5+tdy)*row+start_y),
                             ttxt)

            # Branch weights
            tdy = 0.5*(k+2)
            c.drawString(i*col+start_x+dx, topy-((ny+j*5+tdy)*row+start_y),
                         "w: {:f}".format(snode.nodes[1].text))
            c.setFont("Helvetica", 12+dfontsize)

            if j == 0:
                fx = colx
                fy = cy

        p = c.beginPath()
        p.moveTo(colx-tickx, cy)
        p.lineTo(fx-tickx, fy)
        c.drawPath(p, fill=1)

        i += 1
        if i > 5:
            c.showPage()
            i = 0

    c.showPage()
    c.save()

    print("Generated file {:s}".format(outname))


main.fname = 'Name of the .xml file with the logic tree'
main.outname = 'Name of output .pdf file'

if __name__ == '__main__':
    sap.run(main)
