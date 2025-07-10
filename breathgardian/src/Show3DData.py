import torch
import numpy as np
import vtk

def show3D_ct_vtk(volume, window_level=0.26, window_width=0.005, opacity_max=0.8):
    """
    Prepare a VTK render window and interactor for displaying a 3D CT volume.
    Returns:
        render_window (vtk.vtkRenderWindow): The VTK window
        interactor (vtk.vtkRenderWindowInteractor): The VTK interactor
    """
    # Convert PyTorch tensor to NumPy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Windowing parameters
    min_val = volume.min()
    max_val = volume.max()
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2

    depth, height, width = volume.shape

    # Create VTK image data from NumPy volume
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(width, height, depth)
    vtk_data.SetSpacing(1.0, 1.0, 0.5)  # Z-axis is often scaled
    vtk_data.AllocateScalars(vtk.VTK_FLOAT, 1)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                vtk_data.SetScalarComponentFromFloat(x, y, z, 0, volume[z, y, x])

    # Set up volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_data)
    volume_mapper.SetBlendModeToComposite()
    volume_mapper.SetRequestedRenderModeToGPU()

    # Set volume properties (lighting and interpolation)
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetAmbient(0.3)
    volume_property.SetDiffuse(0.7)
    volume_property.SetSpecular(0.2)
    volume_property.SetSpecularPower(10)

    # Opacity transfer function
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(min_val, 0.0)
    opacity.AddPoint(lower_bound - 0.05 * window_width, 0.0)
    opacity.AddPoint(lower_bound, 0.1)
    opacity.AddPoint(window_level - 0.1 * window_width, opacity_max / 3)
    opacity.AddPoint(window_level, opacity_max / 2)
    opacity.AddPoint(window_level + 0.1 * window_width, opacity_max * 0.7)
    opacity.AddPoint(upper_bound, opacity_max)
    opacity.AddPoint(max_val, opacity_max)
    volume_property.SetScalarOpacity(opacity)

    # Color transfer function
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
    color.AddRGBPoint(lower_bound, 0.2, 0.2, 0.2)
    color.AddRGBPoint(window_level - 0.1 * window_width, 0.4, 0.4, 0.4)
    color.AddRGBPoint(window_level, 0.7, 0.7, 0.7)
    color.AddRGBPoint(window_level + 0.1 * window_width, 0.8, 0.8, 0.8)
    color.AddRGBPoint(upper_bound, 0.9, 0.9, 0.9)
    color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
    volume_property.SetColor(color)

    # Volume actor
    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(volume_property)

    # Renderer setup
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.SetBackground(0.1, 0.1, 0.1)

    # Camera setup
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, -1, 0)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()

    # Render window and interactor setup
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Trackball interaction
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    return render_window, interactor


def show3D_ct_mask_overlay(ct_volume=None, mask_volume=None, vtk_widget=None, 
                          window_level=0.26, window_width=0.005, opacity_max=0.8):
    """
    Show a 3D CT volume with an optional mask overlay using VTK.
    
    Parameters:
    - ct_volume: 3D NumPy or torch tensor
    - mask_volume: 3D mask overlay (same shape as ct_volume)
    - vtk_widget: Optional QVTKRenderWindowInteractor (for PyQt)
    - window_level/window_width: CT contrast controls
    """
    # Convert tensors to NumPy
    if isinstance(ct_volume, torch.Tensor):
        ct_volume = ct_volume.cpu().numpy()
    if mask_volume is not None and isinstance(mask_volume, torch.Tensor):
        mask_volume = mask_volume.cpu().numpy()

    renderer = vtk.vtkRenderer()

    # --- CT Volume Setup ---
    if ct_volume is not None:
        depth, height, width = ct_volume.shape
        spacing_z = height / depth  # Adjust z-spacing

        min_val = ct_volume.min()
        max_val = ct_volume.max()
        lower_bound = window_level - window_width / 2
        upper_bound = window_level + window_width / 2

        # VTK image for CT
        vtk_data_ct = vtk.vtkImageData()
        vtk_data_ct.SetDimensions(width, height, depth)
        vtk_data_ct.SetSpacing(1.0, 1.0, spacing_z)
        vtk_data_ct.AllocateScalars(vtk.VTK_FLOAT, 1)

        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    vtk_data_ct.SetScalarComponentFromFloat(x, y, z, 0, ct_volume[z, y, x])

        # Volume mapper
        ct_mapper = vtk.vtkSmartVolumeMapper()
        ct_mapper.SetInputData(vtk_data_ct)
        ct_mapper.SetBlendModeToComposite()
        ct_mapper.SetRequestedRenderModeToGPU()

        # CT volume properties
        ct_property = vtk.vtkVolumeProperty()
        ct_property.ShadeOn()
        ct_property.SetInterpolationTypeToLinear()
        ct_property.SetAmbient(0.3)
        ct_property.SetDiffuse(0.7)
        ct_property.SetSpecular(0.2)
        ct_property.SetSpecularPower(10)

        # Opacity and color transfer
        ct_opacity = vtk.vtkPiecewiseFunction()
        ct_opacity.AddPoint(min_val, 0.0)
        ct_opacity.AddPoint(lower_bound - 0.05 * window_width, 0.0)
        ct_opacity.AddPoint(lower_bound, 0.1)
        ct_opacity.AddPoint(window_level - 0.1 * window_width, opacity_max / 3)
        ct_opacity.AddPoint(window_level, opacity_max / 2)
        ct_opacity.AddPoint(window_level + 0.1 * window_width, opacity_max * 0.7)
        ct_opacity.AddPoint(upper_bound, opacity_max)
        ct_opacity.AddPoint(max_val, opacity_max)
        ct_property.SetScalarOpacity(ct_opacity)

        ct_color = vtk.vtkColorTransferFunction()
        ct_color.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
        ct_color.AddRGBPoint(lower_bound, 0.2, 0.2, 0.2)
        ct_color.AddRGBPoint(window_level - 0.1 * window_width, 0.4, 0.4, 0.4)
        ct_color.AddRGBPoint(window_level, 0.7, 0.7, 0.7)
        ct_color.AddRGBPoint(window_level + 0.1 * window_width, 0.8, 0.8, 0.8)
        ct_color.AddRGBPoint(upper_bound, 0.9, 0.9, 0.9)
        ct_color.AddRGBPoint(max_val, 1.0, 1.0, 1.0)
        ct_property.SetColor(ct_color)

        # Create actor
        ct_actor = vtk.vtkVolume()
        ct_actor.SetMapper(ct_mapper)
        ct_actor.SetProperty(ct_property)
        renderer.AddVolume(ct_actor)

    # --- Mask Overlay Setup ---
    if mask_volume is not None:
        depth, height, width = mask_volume.shape
        spacing_z = height / depth

        mask_min = mask_volume.min()
        mask_max = mask_volume.max()
        threshold = 0.5 if mask_max <= 1.0 else 0.5 * mask_max

        vtk_data_mask = vtk.vtkImageData()
        vtk_data_mask.SetDimensions(width, height, depth)
        vtk_data_mask.SetSpacing(1.0, 1.0, spacing_z)
        vtk_data_mask.AllocateScalars(vtk.VTK_FLOAT, 1)

        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    vtk_data_mask.SetScalarComponentFromFloat(x, y, z, 0, mask_volume[z, y, x])

        mask_mapper = vtk.vtkSmartVolumeMapper()
        mask_mapper.SetInputData(vtk_data_mask)
        mask_mapper.SetBlendModeToComposite()
        mask_mapper.SetRequestedRenderModeToGPU()

        mask_property = vtk.vtkVolumeProperty()
        mask_property.ShadeOn()
        mask_property.SetInterpolationTypeToLinear()
        mask_property.SetAmbient(0.4)
        mask_property.SetDiffuse(0.6)
        mask_property.SetSpecular(0.2)

        # Opacity function for mask
        mask_opacity = vtk.vtkPiecewiseFunction()
        mask_opacity.AddPoint(0, 0.0)
        mask_opacity.AddPoint(threshold * 0.9, 0.0)
        mask_opacity.AddPoint(threshold, 0.2)
        mask_opacity.AddPoint(threshold * 1.2, 0.5)
        mask_opacity.AddPoint(mask_max, 0.8)
        mask_property.SetScalarOpacity(mask_opacity)

        # Red color map for mask
        mask_color = vtk.vtkColorTransferFunction()
        mask_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
        mask_color.AddRGBPoint(threshold * 0.9, 0.0, 0.0, 0.0)
        mask_color.AddRGBPoint(threshold, 1.0, 0.0, 0.0)
        mask_color.AddRGBPoint(mask_max, 1.0, 0.0, 0.0)
        mask_property.SetColor(mask_color)

        mask_actor = vtk.vtkVolume()
        mask_actor.SetMapper(mask_mapper)
        mask_actor.SetProperty(mask_property)
        renderer.AddVolume(mask_actor)

    # --- Final VTK Setup ---
    renderer.SetBackground(0.1, 0.1, 0.1)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, -1, 0)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()

    if vtk_widget:
        # Render in Qt widget
        render_window = vtk_widget.GetRenderWindow()
        render_window.AddRenderer(renderer)
        interactor = render_window.GetInteractor()
    else:
        # Standalone render
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.Initialize()
        interactor.Start()

    # Apply interaction style and render
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    render_window.Render()
    return render_window, interactor
