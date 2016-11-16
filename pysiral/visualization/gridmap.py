# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 16:18:34 2016

@author: shendric
"""

from pysiral.path import get_module_folder
from pysiral.proj import EASE2North, EASE2South
from pysiral.iotools import get_temp_png_filename
from pysiral.maptools import get_landcoastlines
from pysiral.visualization.mapstyle import GridMapAWIStyle

import os
import numpy as np
from PIL import Image

import shapefile
from pyproj import Proj

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap


class ArcticGridPresentationMap(object):

    def __init__(self):
        self.output = None
        self.temp_file = get_temp_png_filename()
        self.data = None
        self.style = GridMapAWIStyle()
        self.label = GridMapLabels()

    @property
    def projection(self):
        return {"projection": "ortho", "lon_0": 0, "lat_0": 75,
                "resolution": "l"}

    def save2png(self, output):
        self.output = output
        # 1. Create Orthographic map with data plot
        self._create_orthographic_map()
        # 2. Crop and clip Orthographic map, add labels, save
        self._crop_orthographic_map()
        # Remove tempory fils
        self._clean_up()

    def _create_orthographic_map(self):

        # switch off interactive plotting
        plt.ioff()
        figure = plt.figure(**self.style.figure.keyw)
        m = Basemap(**self.projection)

        # load the (shaded) background
        filename = self.style.background.get_filename("north")
        m.warpimage(filename, **self.style.background.keyw)

        # coastline
        if self.style.coastlines.is_active:
            coastlines = get_landcoastlines(m, **self.style.coastlines.keyw)
            plt.gca().add_collection(coastlines)

        # Plot the data as pcolor grid
        data = self.data
        x, y = m(data.pgrid.longitude, data.pgrid.latitude)
        cmap = data.get_cmap()
        m.pcolor(x, y, data.grid, cmap=plt.get_cmap(cmap.name),
                 vmin=cmap.vmin, vmax=cmap.vmax, zorder=110)

        # Plot sea ice concentration as background
        if hasattr(self, "sic"):
            sic = self.sic.grid[:]
            no_ice = sic < 15.
            sic.mask = np.logical_or(sic.mask, no_ice)
            is_ice = np.logical_not(sic.mask)
            sic[np.where(is_ice)] = 90
            plt.pcolormesh(x, y, sic, cmap=plt.get_cmap("gray"),
                           vmin=0, vmax=100, zorder=109)

        # Draw the grid
        # XXX: Skip for noe
        plt.savefig(self.temp_file, dpi=600, facecolor=figure.get_facecolor(),
                    bbox_inches="tight")
        plt.close(figure)

    def _crop_orthographic_map(self):
        # Read the temporary files
        # (only this works, else the image size is capped by screen resolution
        # TODO: try with plt.ioff()
        image = Image.open(self.temp_file)
        imarr = np.array(image)

        # crop the full orthographic image, do not change projections
        x1, x2, y1, y2 = self.style.crop.get_crop_region(image.size)
        cropped_image = imarr[y1:y2, x1:x2, :]

        # Create a new figure
        figure = plt.figure(**self.style.figure.keyw)
        ax = plt.gca()

        # Display cropped image
        plt.axis('off')
        im = ax.imshow(cropped_image)
        ax.set_position([0, 0, 1, 1])

        # clip the image
        if self.style.clip.is_active:
            patch = self.style.clip.get_patch(x1, x2, y1, y2, ax)
            im.set_clip_path(patch)

        # Add labels
        title = self.label.title
        if self.label.annotation != "":
            title = title + " (%s)" % self.label.annotation
        plt.annotate(title, (0.04, 0.93), xycoords="axes fraction",
                     **self.style.font.title)
        plt.annotate(self.label.period, (0.04, 0.89), xycoords="axes fraction",
                     **self.style.font.period)

        # Add colorbar
        cmap = self.data.get_cmap()
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap.name),
                                   norm=plt.Normalize(vmin=cmap.vmin,
                                                      vmax=cmap.vmax))
        sm._A = []
        cb_ax_kwargs = {
            'loc': 3, 'bbox_to_anchor': (0.04, 0.84, 1, 1),
            'width': "30%", 'height': "2%", 'bbox_transform': ax.transAxes,
            'borderpad': 0}
        ticks = MultipleLocator(cmap.step)
        axins = inset_axes(ax, **cb_ax_kwargs)
        cb = plt.colorbar(sm, cax=axins, ticks=ticks, orientation="horizontal")
        cl = plt.getp(cb.ax, 'xmajorticklabels')
        plt.setp(cl, **self.style.font.label)
        parameter_label = self.data.get_label()
        cb.set_label(parameter_label, **self.style.font.label)
        cb.outline.set_linewidth(0.2)
        cb.outline.set_alpha(0.0)
        for t in cb.ax.get_yticklines():
            t.set_color("1.0")
        cb.ax.tick_params('both', length=0.1, which='major', pad=10)
        plt.sca(ax)

        # Add the plane marker at the last point.
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        logo_filename = self.style.logo.get_filename()
        logo = np.array(Image.open(logo_filename))
        im = OffsetImage(logo, **self.style.logo.keyw)
        ab = AnnotationBbox(im, (0.95, 0.89), xycoords='axes fraction',
                            frameon=False, box_alignment=(1, 0))
        ax.add_artist(ab)
        # Now save the map
        plt.savefig(self.output, dpi=self.style.dpi,
                    facecolor=figure.get_facecolor())
        plt.close(figure)

    def _clean_up(self):
        os.remove(self.temp_file)


class AntarcticGridPresentationMap(object):

    def __init__(self):
        self.output = None
        self.temp_file = get_temp_png_filename()
        self.data = None
        self.style = GridMapAWIStyle()
        self.label = GridMapLabels()

    @property
    def projection(self):
        mpl_proj = EASE2South().mpl_projection_keyw
        mpl_proj["width"] = 8500000
        mpl_proj["height"] = 8500000
        return mpl_proj

    def save2png(self, output):
        self.output = output
        # 1. Create Orthographic map with data plot
        self._create_antarctic_map()
        # Remove tempory fils
        # self._clean_up()

    def _create_antarctic_map(self):
        # switch off interactive plotting
        plt.ioff()
        figure = plt.figure(**self.style.figure.keyw)
        ax = plt.gca()
        ax.set_position([0, 0, 1, 1])

        m = Basemap(**self.projection)

        m.drawmapboundary(**self.style.mapboundary.keyw)

        iceshelf, land = get_antarctic_areas()
        proj = EASE2South().projection_keyw
        plot_area(m, land, self.style.continents.keyw["color"], proj)
        plot_area(m, iceshelf, self.style.continents.keyw["iceshelf_color"],
                  proj)

        # load the (shaded) background
        # filename = self.style.background.get_filename("north")
        # m.warpimage(filename, **self.style.background.keyw)
        # coastline
#        if self.style.coastlines.is_active:
#            coastlines = get_landcoastlines(m, **self.style.coastlines.keyw)
#            plt.gca().add_collection(coastlines)
        # Plot the data as pcolor grid

        data = self.data
        x, y = m(data.pgrid.longitude, data.pgrid.latitude)
        cmap = data.get_cmap()
        m.pcolor(x, y, data.grid, cmap=plt.get_cmap(cmap.name),
                 vmin=cmap.vmin, vmax=cmap.vmax, zorder=110)

        if hasattr(self, "sic"):
            sic = self.sic.grid[:]
            no_ice = sic < 15.
            sic.mask = np.logical_or(sic.mask, no_ice)
            is_ice = np.logical_not(sic.mask)
            sic[np.where(is_ice)] = 90
            plt.pcolormesh(x, y, sic, cmap=plt.get_cmap("gray"),
                           vmin=0, vmax=100, zorder=109)

        # Draw the grid
        # XXX: Skip for noe

        # Add labels
        title = self.label.title
        if self.label.annotation != "":
            title = title + " (%s)" % self.label.annotation
        plt.annotate(title, (0.04, 0.93), xycoords="axes fraction",
                     **self.style.font.title)
        plt.annotate(self.label.period, (0.04, 0.89), xycoords="axes fraction",
                     **self.style.font.period)
        # Add colorbar
        cmap = self.data.get_cmap()
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap.name),
                                   norm=plt.Normalize(vmin=cmap.vmin,
                                                      vmax=cmap.vmax))
        sm._A = []
        cb_ax_kwargs = {
            'loc': 3, 'bbox_to_anchor': (0.04, 0.10, 1, 1),
            'width': "30%", 'height': "2%", 'bbox_transform': ax.transAxes,
            'borderpad': 0}
        ticks = MultipleLocator(cmap.step)
        axins = inset_axes(ax, **cb_ax_kwargs)
        cb = plt.colorbar(sm, cax=axins, ticks=ticks, orientation="horizontal")
        cl = plt.getp(cb.ax, 'xmajorticklabels')
        plt.setp(cl, **self.style.font.label)
        parameter_label = self.data.get_label()
        cb.set_label(parameter_label, **self.style.font.label)
        cb.outline.set_linewidth(0.2)
        cb.outline.set_alpha(0.0)
        for t in cb.ax.get_yticklines():
            t.set_color("1.0")
        cb.ax.tick_params('both', length=0.1, which='major', pad=10)
        plt.sca(ax)

        # Add the plane marker at the last point.
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        logo_filename = self.style.logo.get_filename()
        logo = np.array(Image.open(logo_filename))
        im = OffsetImage(logo, **self.style.logo.keyw)
        ab = AnnotationBbox(im, (0.95, 0.89), xycoords='axes fraction',
                            frameon=False, box_alignment=(1, 0))
        ax.add_artist(ab)

        plt.savefig(self.output, dpi=self.style.dpi,
                    facecolor=figure.get_facecolor())
        plt.close(figure)

#    def _crop_orthographic_map(self):
#        # Read the temporary files
#        # (only this works, else the image size is capped by screen resolution
#        # TODO: try with plt.ioff()
#        image = Image.open(self.temp_file)
#        imarr = np.array(image)
#        # crop the full orthographic image, do not change projections
#        x1, x2, y1, y2 = self.style.crop.get_crop_region(image.size)
#        cropped_image = imarr[y1:y2, x1:x2, :]
#        # Create a new figure
#        figure = plt.figure(**self.style.figure.keyw)
#        ax = plt.gca()
#        # Display cropped image
#        plt.axis('off')
#        im = ax.imshow(cropped_image)
#        ax.set_position([0, 0, 1, 1])
#        # clip the image
#        if self.style.clip.is_active:
#            patch = self.style.clip.get_patch(x1, x2, y1, y2, ax)
#            im.set_clip_path(patch)
#        # Add labels
#        title = self.label.title
#        if self.label.annotation != "":
#            title = title + " (%s)" % self.label.annotation
#        plt.annotate(title, (0.04, 0.93), xycoords="axes fraction",
#                     **self.style.font.title)
#        plt.annotate(self.label.period, (0.04, 0.89), xycoords="axes fraction",
#                     **self.style.font.period)
#        # Add colorbar
#        cmap = self.data.get_cmap()
#        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap.name),
#                                   norm=plt.Normalize(vmin=cmap.vmin,
#                                                      vmax=cmap.vmax))
#        sm._A = []
#        cb_ax_kwargs = {
#            'loc': 3, 'bbox_to_anchor': (0.04, 0.84, 1, 1),
#            'width': "30%", 'height': "2%", 'bbox_transform': ax.transAxes,
#            'borderpad': 0}
#        ticks = MultipleLocator(cmap.step)
#        axins = inset_axes(ax, **cb_ax_kwargs)
#        cb = plt.colorbar(sm, cax=axins, ticks=ticks, orientation="horizontal")
#        cl = plt.getp(cb.ax, 'xmajorticklabels')
#        plt.setp(cl, **self.style.font.label)
#        parameter_label = self.data.get_label()
#        cb.set_label(parameter_label, **self.style.font.label)
#        cb.outline.set_linewidth(0.2)
#        cb.outline.set_alpha(0.0)
#        for t in cb.ax.get_yticklines():
#            t.set_color("1.0")
#        cb.ax.tick_params('both', length=0.1, which='major', pad=10)
#        plt.sca(ax)
#
#        # Add the plane marker at the last point.
#        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#        logo_filename = self.style.logo.get_filename()
#        logo = np.array(Image.open(logo_filename))
#        im = OffsetImage(logo, **self.style.logo.keyw)
#        ab = AnnotationBbox(im, (0.95, 0.89), xycoords='axes fraction',
#                            frameon=False, box_alignment=(1, 0))
#        ax.add_artist(ab)
#        # Now save the map
#        plt.savefig(self.output, dpi=self.style.dpi,
#                    facecolor=figure.get_facecolor())
#        plt.close(figure)
#
#    def _clean_up(self):
#        os.remove(self.temp_file)


class ArcticGridPaperMap(object):

    def __init__(self):
        self.output = None
        self.temp_file = get_temp_png_filename()
        self.data = None
        self.style = GridMapAWIStyle()
        self.label = GridMapLabels()

    @property
    def projection(self):
        return {"width": 6e6, "height": 4.75e6, "lon_0": -45, "lat_0": 90,
                "lat_ts": 88, "projection": "laea", "resolution": "i"}

    def save2png(self, output):
        self.output = output
        # 1. Create Orthographic map with data plot
        self._create_map()

    def _create_map(self):

        # switch off interactive plotting
        plt.ioff()
        figure = plt.figure(**self.style.figure.keyw)

        # Basemap settings
        m = Basemap(**self.projection)
        m.drawmapboundary(**self.style.mapboundary.keyw)
        m.fillcontinents(zorder=120, **self.style.continents.keyw)
        if self.style.coastlines.is_active:
            coastlines = get_landcoastlines(m, **self.style.coastlines.keyw)
            coastlines.set_zorder(120)
            plt.gca().add_collection(coastlines)

        # Plot the data as pcolor grid
        data = self.data
        x, y = m(data.pgrid.longitude, data.pgrid.latitude)
        cmap = data.get_cmap()
        m.pcolor(x, y, data.grid, cmap=plt.get_cmap(cmap.name),
                 vmin=cmap.vmin, vmax=cmap.vmax, zorder=110, lw=0.1,
                 edgecolors=self.style.mapboundary.keyw["fill_color"])

        # Annotation
        plt.annotate(self.label.annotation, (0.98, 0.93), ha="right",
                     xycoords="axes fraction", zorder=130,
                     **self.style.font.annotation)

        # Save plot
        plt.savefig(self.output, dpi=600, facecolor=figure.get_facecolor(),
                    bbox_inches="tight")
        plt.close(figure)

    def _clean_up(self):
        os.remove(self.temp_file)


class AntarcticGridPaperMap(object):

    def __init__(self):
        self.output = None
        self.temp_file = get_temp_png_filename()
        self.data = None
        self.style = GridMapAWIStyle()
        self.label = GridMapLabels()

    @property
    def projection(self):
        return {"width": 6e6, "height": 4.75e6, "lon_0": -45, "lat_0": 90,
                "lat_ts": 88, "projection": "laea", "resolution": "i"}

    def save2png(self, output):
        self.output = output
        # 1. Create Orthographic map with data plot
        self._create_map()

    def _create_map(self):

        # switch off interactive plotting
        plt.ioff()
        figure = plt.figure(**self.style.figure.keyw)

        # Basemap settings
        m = Basemap(**self.projection)
        m.drawmapboundary(**self.style.mapboundary.keyw)
        m.fillcontinents(zorder=120, **self.style.continents.keyw)
        if self.style.coastlines.is_active:
            coastlines = get_landcoastlines(m, **self.style.coastlines.keyw)
            coastlines.set_zorder(120)
            plt.gca().add_collection(coastlines)

        # Plot the data as pcolor grid
        data = self.data
        x, y = m(data.pgrid.longitude, data.pgrid.latitude)
        cmap = data.get_cmap()
        m.pcolor(x, y, data.grid, cmap=plt.get_cmap(cmap.name),
                 vmin=cmap.vmin, vmax=cmap.vmax, zorder=110, lw=0.1,
                 edgecolors=self.style.mapboundary.keyw["fill_color"])

        # Annotation
        plt.annotate(self.label.annotation, (0.98, 0.93), ha="right",
                     xycoords="axes fraction", zorder=130,
                     **self.style.font.annotation)

        # Save plot
        plt.savefig(self.output, dpi=600, facecolor=figure.get_facecolor(),
                    bbox_inches="tight")
        plt.close(figure)

    def _clean_up(self):
        os.remove(self.temp_file)


class GridMapLabels(object):

    def __init__(self):
        self.title = ""
        self.period = ""
        self.annotation = ""
        self.copyright = ""


def get_antarctic_areas():
    folder = get_module_folder(__file__)
    ant_shape_file = os.path.join(folder, "shapes",
                                  "antarctic_cst10_polygon.shp")
    sf = shapefile.Reader(ant_shape_file)
    shapes = sf.shapes()
    iceshelf, land = [], []
    for i, record in enumerate(sf.records()):
        if record[1] == "land":
            land.append(shapes[i])
            continue
        if record[1] == "iceshelf":
            iceshelf.append(shapes[i])
    return iceshelf, land


def plot_area(basemap, areas, color, projection):
    color_fill, color_edge = color, color
    for j, area in enumerate(areas):
        points = np.array(area.points)
        area.parts.append(len(points)-1)
        for i in np.arange(len(area.parts)-1):
            i0, i1 = area.parts[i], area.parts[i+1]
            prjx, prjy = points[i0:i1, 0], points[i0:i1, 1]
            p = Proj(**projection)
            lons, lats = p(prjx, prjy, inverse=True)
            x, y = basemap(lons, lats)
            if x[0]-x[-1] > 0:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            plt.fill(x, y, color=color_fill, lw=0.25, zorder=1000)
            plt.plot(x, y, color=color_edge, lw=0.25, zorder=1000)
