#!/usr/bin/env python
# coding: utf-8

import re

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from reportlab.lib.colors import red, black, blue, grey
from openquake.baselib import sap
from openquake.hazardlib.nrml import to_python


def main(fname, *, outname='report_lt.pdf'):

    lt = to_python(fname)

    row = 15
    col = 150

    start_x = 30
    start_y = 30
    dx = 5
    ny = 10
    c = canvas.Canvas(outname, pagesize=(A4[1], A4[0]))

    c.setFont("Helvetica-Bold", 14)
    tmp = "Logic Tree ID: {:s}".format(lt.attrib['logicTreeID'])
    c.drawString(start_x, A4[0]-start_y, tmp)
    c.setFont("Helvetica", 8)
    c.setFillColor(grey)
    c.drawString(start_x, A4[0]-start_y-13, "File: {:s}".format(fname))

    # Column counter
    i = 0
    tickx = 10
    tickix = 2
    for n, node in enumerate(lt.nodes):

        c.setFont("Helvetica", 12)
        c.setFillColor(red)
        c.drawString(i*col+start_x, A4[0]-(3*row+start_y),
                     "[{:d}] BSet".format(n))
        c.setFillColor(blue)
        c.drawString(i*col+start_x+50, A4[0]-(3*row+start_y),
                     "{:s}".format(node.attrib['branchSetID']))
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawString(i*col+start_x, A4[0]-(4*row+start_y),
                     "u: {:s}".format(node.attrib['uncertaintyType']))

        cnt = 1
        for key in node.attrib:
            if re.search('^apply', key):
                c.drawString(i*col+start_x, A4[0]-((4+cnt)*row+start_y),
                             "a: {:s}".format(key))
                c.drawString(i*col+start_x, A4[0]-((4+cnt+1)*row+start_y),
                             "   {:s}".format(node.attrib[key]))
                cnt += 2

        c.setFont("Helvetica", 12)

        for j, snode in enumerate(node.nodes):
            c.setFillColor(red)
            cx = i*col+start_x
            cy = A4[0]-((ny+j*3)*row+start_y)
            c.drawString(cx, cy, "Branch")
            c.setFillColor(blue)
            c.drawString(i*col+start_x+40, A4[0]-((ny+j*3)*row+start_y),
                         "{:s}".format(snode.attrib['branchID']))

            c.setFillColor(black)
            p = c.beginPath()
            p.moveTo(cx-tickix, cy)
            p.lineTo(cx-tickx, cy)
            c.drawPath(p, fill=1)

            c.setFont("Helvetica", 10)
            c.drawString(i*col+start_x+dx, A4[0]-((ny+j*3+1)*row+start_y),
                         "v: {:s}".format(snode.nodes[0].text))
            c.drawString(i*col+start_x+dx, A4[0]-((ny+j*3+2)*row+start_y),
                         "w: {:f}".format(snode.nodes[1].text))
            c.setFont("Helvetica", 12)

            if j == 0:
                fx = cx
                fy = cy

        p = c.beginPath()
        p.moveTo(cx-tickx, cy)
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
