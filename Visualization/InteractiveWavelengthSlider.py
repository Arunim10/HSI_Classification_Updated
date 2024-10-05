import pandas as pd
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from glob import glob
import h5py

hdr_files = list(glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\HDR_Data\\Extracted_Folder\\*\\*.hdr.h5"))
wavelengths = h5py.File(hdr_files[0],'r')['wavelengths']

df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector
from ipywidgets import interact, interactive, fixed
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

# class InteractiveWavelengthSlider:
#     def __init__(self, hsi_data, wavelengths):
#         self.hsi_data = hsi_data
#         self.wavelengths = wavelengths
#         self.current_wavelength_index = 0

#         self.fig = plt.figure(figsize=(15, 8))
#         self.ax_image = plt.subplot2grid((1, 2), (0, 0))
#         self.ax_spectrum = plt.subplot2grid((1, 2), (0, 1))
        
#         plt.subplots_adjust(bottom=0.2, wspace=0.2)

#         # Initialize image display
#         self.im = self.ax_image.imshow(self.hsi_data[:, :, self.current_wavelength_index], cmap='Spectral')
#         self.ax_image.set_title(f'Wavelength: {self.wavelengths[self.current_wavelength_index]:.2f} nm')

#         # Initialize spectral profile
#         self.mean_spectrum = np.mean(self.hsi_data.reshape(-1, self.hsi_data.shape[2]), axis=0)
#         self.spectrum_line, = self.ax_spectrum.plot(self.wavelengths, self.mean_spectrum)
#         self.wavelength_indicator = self.ax_spectrum.axvline(x=self.wavelengths[self.current_wavelength_index], color='r')
#         self.ax_spectrum.set_xlabel('Wavelength (nm)')
#         self.ax_spectrum.set_ylabel('Reflectance')
#         self.ax_spectrum.set_title('Mean Spectral Profile')

#         # Create wavelength slider
#         self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
#         self.slider = Slider(self.ax_slider, 'Wavelength', 0, len(self.wavelengths) - 1,
#                              valinit=self.current_wavelength_index, valstep=1)
#         self.slider.on_changed(self.update_wavelength)

#         # Set up rectangle selector for pixel selection
#         self.rect_selector = RectangleSelector(self.ax_image, self.on_select, useblit=True,
#                                                button=[1], minspanx=5, minspany=5,
#                                                spancoords='pixels', interactive=True)

#     def update_wavelength(self, val):
#         self.current_wavelength_index = int(val)
#         self.im.set_data(self.hsi_data[:, :, self.current_wavelength_index])
#         self.ax_image.set_title(f'Wavelength: {self.wavelengths[self.current_wavelength_index]:.2f} nm')
#         self.wavelength_indicator.set_xdata(self.wavelengths[self.current_wavelength_index])
#         self.fig.canvas.draw_idle()

#     def on_select(self, eclick, erelease):
#         x1, y1 = int(eclick.xdata), int(eclick.ydata)
#         x2, y2 = int(erelease.xdata), int(erelease.ydata)
#         region = self.hsi_data[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2), :]
#         mean_spectrum = np.mean(region, axis=(0, 1))
#         self.spectrum_line.set_ydata(mean_spectrum)
#         self.ax_spectrum.relim()
#         self.ax_spectrum.autoscale_view()
#         self.fig.canvas.draw_idle()

#     def show(self):
#         plt.show()      

# import itertools

# class InteractiveWavelengthSlider:
#     def __init__(self, hsi_data, wavelengths):
#         self.hsi_data = hsi_data
#         self.wavelengths = wavelengths
#         self.current_wavelength_index = 0

#         # Create figure and axes
#         self.fig = plt.figure(figsize=(15, 8))
#         self.ax_image = plt.subplot2grid((1, 2), (0, 0))
#         self.ax_spectrum = plt.subplot2grid((1, 2), (0, 1))
        
#         plt.subplots_adjust(bottom=0.2, wspace=0.2)

#         # Initialize image display
#         self.im = self.ax_image.imshow(self.hsi_data[:, :, self.current_wavelength_index], cmap='Spectral')
#         self.ax_image.set_title(f'Wavelength: {self.wavelengths[self.current_wavelength_index]:.2f} nm')

#         # Initialize spectral profile
#         self.mean_spectrum = np.mean(self.hsi_data.reshape(-1, self.hsi_data.shape[2]), axis=0)
#         self.spectrum_lines = []
#         self.rectangles = []  # Keep track of drawn rectangles
#         self.wavelength_indicator = self.ax_spectrum.axvline(x=self.wavelengths[self.current_wavelength_index], color='r')
#         self.ax_spectrum.set_xlabel('Wavelength (nm)')
#         self.ax_spectrum.set_ylabel('Reflectance')
#         self.ax_spectrum.set_title('Mean Spectral Profile')

#         # Color iterator for different regions
#         self.color_cycle = itertools.cycle(plt.cm.get_cmap('tab10').colors)  # Use tab10 colormap for distinct colors
        
#         # Create wavelength slider
#         self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
#         self.slider = Slider(self.ax_slider, 'Wavelength', 0, len(self.wavelengths) - 1,
#                              valinit=self.current_wavelength_index, valstep=1)
#         self.slider.on_changed(self.update_wavelength)

#         # Set up rectangle selector for pixel selection
#         self.rect_selector = RectangleSelector(self.ax_image, self.on_select, useblit=True,
#                                                button=[1], minspanx=5, minspany=5,
#                                                spancoords='pixels', interactive=True)

#     def update_wavelength(self, val):
#         self.current_wavelength_index = int(val)
#         self.im.set_data(self.hsi_data[:, :, self.current_wavelength_index])
#         self.ax_image.set_title(f'Wavelength: {self.wavelengths[self.current_wavelength_index]:.2f} nm')
#         self.wavelength_indicator.set_xdata(self.wavelengths[self.current_wavelength_index])
#         self.fig.canvas.draw_idle()

#     def on_select(self, eclick, erelease):
#         # Get coordinates of the selected region
#         x1, y1 = int(eclick.xdata), int(eclick.ydata)
#         x2, y2 = int(erelease.xdata), int(erelease.ydata)

#         # Extract the region and compute its mean spectrum
#         region = self.hsi_data[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2), :]
#         mean_spectrum = np.mean(region, axis=(0, 1))

#         # Plot the mean spectrum of the selected region with a unique color
#         color = next(self.color_cycle)
#         spectrum_line, = self.ax_spectrum.plot(self.wavelengths, mean_spectrum, label=f'Region {len(self.spectrum_lines)+1}', color=color)
#         self.spectrum_lines.append(spectrum_line)

#         # Add a rectangle to the image to visualize the selected region
#         rect = Rectangle((min(x1, x2), min(y1, y2)), abs(x2 - x1), abs(y2 - y1), linewidth=2, edgecolor=color, facecolor='none')
#         self.ax_image.add_patch(rect)
#         self.rectangles.append(rect)

#         # Adjust the plot to show the legend for multiple regions
#         self.ax_spectrum.relim()
#         self.ax_spectrum.autoscale_view()
#         self.ax_spectrum.legend()
#         self.fig.canvas.draw_idle()

#     def show(self):
#         plt.show()

class InteractiveWavelengthSlider:
    def __init__(self, hsi_data, wavelengths):
        self.hsi_data = hsi_data
        self.wavelengths = wavelengths
        self.current_wavelength_index = 0
        self.selected_regions = []
        self.color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
        self.current_color_index = 0

        self.fig, (self.ax_image, self.ax_spectrum) = plt.subplots(1, 2, figsize=(15, 8))
        plt.subplots_adjust(bottom=0.2)

        # Initialize image display
        self.im = self.ax_image.imshow(self.hsi_data[:, :, self.current_wavelength_index], cmap='viridis')
        self.ax_image.set_title(f'Wavelength: {self.wavelengths[self.current_wavelength_index]:.2f} nm')

        # Initialize spectral profile
        self.mean_spectrum = np.mean(self.hsi_data.reshape(-1, self.hsi_data.shape[2]), axis=0)
        self.spectrum_line, = self.ax_spectrum.plot(self.wavelengths, self.mean_spectrum, color='gray', alpha=0.5, label='Overall Mean')
        self.wavelength_indicator = self.ax_spectrum.axvline(x=self.wavelengths[self.current_wavelength_index], color='r')
        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Reflectance')
        self.ax_spectrum.set_title('Spectral Profiles')
        self.ax_spectrum.legend()

        # Create wavelength slider
        self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(self.ax_slider, 'Wavelength', 0, len(self.wavelengths) - 1, valinit=self.current_wavelength_index, valstep=1)
        self.slider.on_changed(self.update_wavelength)

        # Set up mouse event handling for region selection
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.pressed = False
        self.rect = None
        self.start_point = None

        # Add a clear button
        self.ax_clear = plt.axes([0.8, 0.01, 0.1, 0.04])
        self.clear_button = Button(self.ax_clear, 'Clear')
        self.clear_button.on_clicked(self.clear_selections)

    def update_wavelength(self, val):
        self.current_wavelength_index = int(val)
        self.im.set_data(self.hsi_data[:, :, self.current_wavelength_index])
        self.ax_image.set_title(f'Wavelength: {self.wavelengths[self.current_wavelength_index]:.2f} nm')
        self.wavelength_indicator.set_xdata(self.wavelengths[self.current_wavelength_index])
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax_image:
            return
        self.pressed = True
        self.start_point = (event.xdata, event.ydata)
        self.rect = Rectangle(self.start_point, 0, 0, fill=False, edgecolor=self.color_cycle[self.current_color_index])
        self.ax_image.add_patch(self.rect)

    def on_motion(self, event):
        if self.pressed and event.inaxes == self.ax_image:
            x0, y0 = self.start_point
            x1, y1 = event.xdata, event.ydata
            self.rect.set_width(x1 - x0)
            self.rect.set_height(y1 - y0)
            self.rect.set_xy((min(x0, x1), min(y0, y1)))
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.pressed and event.inaxes == self.ax_image:
            self.pressed = False
            x0, y0 = self.start_point
            x1, y1 = event.xdata, event.ydata
            
            x_min, x_max = int(min(x0, x1)), int(max(x0, x1))
            y_min, y_max = int(min(y0, y1)), int(max(y0, y1))
            
            region = self.hsi_data[y_min:y_max, x_min:x_max, :]
            mean_spectrum = np.mean(region, axis=(0, 1))

            color = self.color_cycle[self.current_color_index]
            self.current_color_index = (self.current_color_index + 1) % len(self.color_cycle)

            line, = self.ax_spectrum.plot(self.wavelengths, mean_spectrum, color=color, label=f'Region {len(self.selected_regions) + 1}')
            self.selected_regions.append((line, self.rect))
            
            self.ax_spectrum.legend()
            self.ax_spectrum.relim()
            self.ax_spectrum.autoscale_view()
            self.fig.canvas.draw_idle()

    def clear_selections(self, event):
        for line, rect in self.selected_regions:
            line.remove()
            rect.remove()
        self.selected_regions.clear()
        self.current_color_index = 0
        self.ax_spectrum.legend()
        self.ax_spectrum.relim()
        self.ax_spectrum.autoscale_view()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

import albumentations as A
# Example usage:
# Assuming you have your HSI data and wavelengths:
id = 20
hsi = np.transpose(loadmat(df.loc[id,'HSI_Path'])['HDR'],axes=(1,2,0))
mask = loadmat(df.loc[id,'MASK_Path'])['MASK']
tf = A.Compose([
    # A.Rotate(limit=90, p=1.0)
    # A.HorizontalFlip(p=1)   
    A.VerticalFlip(p=1)
    # A.RandomRotate90(p=1)
    # A.Transpose(p=1)
    # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    # A.GridDistortion(p=1)
    # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
    # A.RandomSizedCrop(min_max_height=(100,200), height=hsi.shape[0], width=hsi.shape[1], p=1)
])
aug = tf(image=hsi,mask=mask)
cmap = 'viridis'
aug_img = aug['image']
aug_mask = aug['mask']
# hsi_data = np.random.rand(320, 410, 256)  # Replace with your actual HSI data
# wavelengths = np.linspace(400, 2500, 256)  # Replace with your actual wavelengths
slider = InteractiveWavelengthSlider(aug_img, np.array(wavelengths))
slider.show()

