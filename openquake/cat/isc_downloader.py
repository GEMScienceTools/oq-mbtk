# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2016 GEM Foundation
#
# The Catalogue Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# with this download. If not, see <http://www.gnu.org/licenses/>
#

"""
Utility to download the ISC catalogue from website.
"""
import sys
import os
import time
import collections

# Python 2 & 3 compatible
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request


# Python 3 urllib returns download as bytearray, so this decodes it
def parse_page(page):
    if sys.version_info.major < 3:
        return page
    else:
        return page.decode("utf-8")


class ISCBulletinUrl():

    def __init__(self):

        # DEFAULT VALUES
        self.BaseServer = "http://www.isc.ac.uk/cgi-bin/web-db-v4?"

        self.Request = collections.OrderedDict()

        # Compulsory fields
        self.Request["CatalogueType"]           = "request=REVIEWED"
        self.Request["OutputFormat"]            = "out_format=ISF"
        self.Request["SearchAreaShape"]         = "searchshape=RECT"
        self.Request["RectangleBottomLatitude"] = "bot_lat=36"
        self.Request["RectangleTopLatitude"]    = "top_lat=48"
        self.Request["RectangleLeftLongitude"]  = "left_lon=6"
        self.Request["RectangleRightLongitude"] = "right_lon=19"
        self.Request["CircularLatitude"]        = "ctr_lat="
        self.Request["CircularLongitude"]       = "ctr_lon="
        self.Request["CircularRadius"]          = "radius="
        self.Request["MaxDistanceUnits"]        = "max_dist_units=deg"
        self.Request["SeismicRegionNumber"]     = "srn="
        self.Request["GeogrephicRegionNumber"]  = "grn="
        self.Request["PolygonCoordinates"]      = "coordvals="
        self.Request["StartYear"]               = "start_year=2012"
        self.Request["StartMonth"]              = "start_month=01"
        self.Request["StartDay"]                = "start_day=01"
        self.Request["StartTime"]               = "start_time=00:00:00"
        self.Request["EndYear"]                 = "end_year=2013"
        self.Request["EndMonth"]                = "end_month=12"
        self.Request["EndDay"]                  = "end_day=31"
        self.Request["EndTime"]                 = "end_time=23:59:59"
        self.Request["MinimumDepth"]            = "min_dep="
        self.Request["MaximumDepth"]            = "max_dep="
        self.Request["NoDepthEvents"]           = "null_dep=on"
        self.Request["MinimumMagnitude"]        = "min_mag="
        self.Request["MaximumMagnitude"]        = "max_mag="
        self.Request["NoMagnitudeEvents"]       = "null_mag=on"
        self.Request["MagnitudeType"]           = "req_mag_type="
        self.Request["MagnitudeAgency"]         = "req_mag_agcy="
        self.Request["FocalMechanismAgency"]    = "req_fm_agcy=Any"

        # Optional Fields
        self.Request["IncludePhases"]           = "include_phases=off"
        self.Request["MinimumPhaseNumber"]      = "min_def="
        self.Request["MaximumPhaseNumber"]      = "max_def="
        self.Request["NoKnownPhases"]           = "null_phs="
        self.Request["PrimeOnly"]               = "prime_only="
        self.Request["IncludeMagnitudes"]       = "include_magnitudes=on"
        self.Request["IncludeHeaders"]          = "include_headers=on"
        self.Request["IncludeComments"]         = "include_comments=off"
        self.Request["IncludeLinks"]            = "include_links=off"

    def UseMirror(self):
        tmp = "http://isc-mirror.iris.washington.edu/cgi-bin/web-db-v4?"
        self.BaseServer = tmp

    def ListFields(self):
        print("\nCURRENT SETTINGS:\n")
        for Key in self.Request:
            Value = self.Request[Key].split("=")[1]
            if not Value:
                Value = "[Empty]"
            print("\t" + Key + " = " + Value)

    def SetField(self, field_name, field_value):
        """
        """
        buf = self.Request[field_name]
        buf = buf.split("=")[0]
        self.Request[field_name] = buf + "=%s" % field_value

    def SaveSettings(self, ParamFile):
        """
        """
        ParFile = open(ParamFile, "w")
        for Key in self.Request:
            Value = self.Request[Key].split("=")[1]
            if not Value:
                Value = "Null"
            ParFile.write("%s=%s" % (Key, Value))
            if Key != list(self.Request.keys())[-1]:
                ParFile.write("\n")
        ParFile.close()

    def LoadSettings(self, ParamFile):
        ParFile = open(ParamFile, "r")
        for Line in ParFile:
            Key = Line.split("=")[0]
            Value = Line.split("=")[1].strip('\n')
            if Value == "Null":
                Value = ""
            self.SetField(Key, Value)
        ParFile.close()

    def SetSearchArea(self, Lat, Lon):
        self.SetField("SearchAreaShape", "RECT")
        self.SetField("RectangleBottomLatitude", str(Lat[0]))
        self.SetField("RectangleTopLatitude", str(Lat[1]))
        self.SetField("RectangleLeftLongitude", str(Lon[0]))
        self.SetField("RectangleRightLongitude", str(Lon[1]))

    def SetSearchTime(self, Year0, Year1):
        self.SetField("StartYear", str(Year0))
        self.SetField("EndYear", str(Year1))

    def CreateUrl(self):
        UrlString = self.BaseServer
        for value in self.Request.values():
            UrlString += value + "&"
        return UrlString

    def DownloadBlock(self):

        Tries = 0
        CatBlock = ""
        UrlString = self.CreateUrl()

        while True:
            UrlReq = Request(UrlString)
            UrlRes = urlopen(UrlReq)
            Page = parse_page(UrlRes.read())
            UrlRes.close()
            if Page.find("Sorry") > -1:
                if Tries > 10:
                    print("Warning: Maximum number of attempts reached...")
                    break
                print("Warning: Server is busy, retrying in a few seconds...")
                time.sleep(30)
                Tries += 1
            else:
                CatStart = Page.find("DATA_TYPE")
                CatStop = Page.find("STOP")
                if CatStart > -1 and CatStop > -1:
                    CatBlock = Page[CatStart:CatStop-1]
                    break
                else:
                    msg = "Catalogue not available for the selected period."
                    print(msg)
                    break

        return CatBlock

    def GetCatalogue(self, SplitYears=0):
        """
        :param SplitYears
        """

        accumulator = ""
        if SplitYears < 1:
            accumulator = self.DownloadBlock()
        else:
            # Make sure the whole year is covered
            self.SetField("StartMonth", "01")
            self.SetField("StartDay", "01")
            self.SetField("StartTime", "00:00:00")
            self.SetField("EndMonth", "12")
            self.SetField("EndDay", "31")
            self.SetField("EndTime", "23:59:59")

            # Split download into several chunks
            StartYear = int(self.Request["StartYear"].split("=")[1])
            EndYear = int(self.Request["EndYear"].split("=")[1])

            for SY in range(StartYear, EndYear+1, SplitYears):

                EY = min([EndYear, SY+SplitYears-1])
                self.SetField("StartYear", SY)
                self.SetField("EndYear", EY)

                print("Downloading block:", SY, "-", EY)
                Chunk = self.DownloadBlock()

                if SY != StartYear:
                    # Remove header from data blocks
                    Chunk = Chunk.split('\n', 2)[-1]

                accumulator += Chunk
        self.CatBlock = accumulator

    def WriteOutput(self, OutputFile, OverWrite=False):

        if os.path.isfile(OutputFile) and not OverWrite:
            print("Warning: File exists. Use OverWrite option.")
            return

        try:
            with open(OutputFile, "w") as CatFile:
                CatFile.write("%s" % self.CatBlock)
                CatFile.close()
        except:
            print("Warning: Cannot open output file....")
