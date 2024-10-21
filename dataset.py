import numpy as np
import cv2
import os
from scipy.ndimage import map_coordinates
from scipy.io import savemat
from scipy.interpolate import griddata
from tqdm import tqdm  # Progress bar library

# === FEM-Based Deformation Functions ===

height, width = 256, 256

# Generate mesh
def generate_mesh(num_elem_x, num_elem_y, width, height):
    num_nodes_x = num_elem_x * 2 + 1  # Quadratic elements
    num_nodes_y = num_elem_y * 2 + 1

    node_x = np.linspace(0, width - 1, num_nodes_x)
    node_y = np.linspace(0, height - 1, num_nodes_y)
    node_X, node_Y = np.meshgrid(node_x, node_y)
    node_coords = np.column_stack((node_X.ravel(), node_Y.ravel()))

    elements = []
    for i in range(num_elem_y):
        for j in range(num_elem_x):
            n1 = (2 * i) * num_nodes_x + 2 * j  # (-1, -1)
            n2 = n1 + 2                          # (1, -1)
            n3 = n1 + 2 + 2 * num_nodes_x        # (1, 1)
            n4 = n1 + 0 + 2 * num_nodes_x        # (-1, 1)
            n5 = n1 + 1                          # (0, -1)
            n6 = n1 + 2 + num_nodes_x            # (1, 0)
            n7 = n1 + 1 + 2 * num_nodes_x        # (0, 1)
            n8 = n1 + 0 + num_nodes_x            # (-1, 0)
            n9 = n1 + 1 + num_nodes_x            # (0, 0)
            element_nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9]
            elements.append(element_nodes)
    return node_coords, elements

# Define shape functions
def shape_functions(xi, eta):
    N = np.zeros(9)
    N[0] = 0.25 * (xi - 1) * (eta - 1) * xi * eta
    N[1] = 0.5 * (1 - xi ** 2) * (eta - 1) * eta
    N[2] = 0.25 * (xi + 1) * (eta - 1) * xi * eta
    N[3] = 0.5 * (xi + 1) * xi * (1 - eta ** 2)
    N[4] = 0.25 * (xi + 1) * (eta + 1) * xi * eta
    N[5] = 0.5 * (1 - xi ** 2) * (eta + 1) * eta
    N[6] = 0.25 * (xi - 1) * (eta + 1) * xi * eta
    N[7] = 0.5 * (xi - 1) * xi * (1 - eta ** 2)
    N[8] = (1 - xi ** 2) * (1 - eta ** 2)
    return N

# Define shape function derivatives
def shape_function_derivatives(xi, eta):
    dN_dxi = np.zeros(9)
    dN_deta = np.zeros(9)
    # Derivatives with respect to xi
    dN_dxi[0] = 0.25 * (2 * xi - 1) * (eta - 1) * eta
    dN_dxi[1] = -xi * (eta - 1) * eta
    dN_dxi[2] = 0.25 * (2 * xi + 1) * (eta - 1) * eta
    dN_dxi[3] = 0.5 * (2 * xi + 1) * (1 - eta ** 2)
    dN_dxi[4] = 0.25 * (2 * xi + 1) * (eta + 1) * eta
    dN_dxi[5] = -xi * (eta + 1) * eta
    dN_dxi[6] = 0.25 * (2 * xi - 1) * (eta + 1) * eta
    dN_dxi[7] = 0.5 * (2 * xi - 1) * (1 - eta ** 2)
    dN_dxi[8] = -2 * xi * (1 - eta ** 2)
    # Derivatives with respect to eta
    dN_deta[0] = 0.25 * (xi - 1) * (2 * eta - 1) * xi
    dN_deta[1] = 0.5 * (1 - xi ** 2) * (2 * eta - 1)
    dN_deta[2] = 0.25 * (xi + 1) * (2 * eta - 1) * xi
    dN_deta[3] = -xi * (xi + 1) * eta
    dN_deta[4] = 0.25 * (xi + 1) * (2 * eta + 1) * xi
    dN_deta[5] = 0.5 * (1 - xi ** 2) * (2 * eta + 1)
    dN_deta[6] = 0.25 * (xi - 1) * (2 * eta + 1) * xi
    dN_deta[7] = -xi * (xi - 1) * eta
    dN_deta[8] = -2 * eta * (1 - xi ** 2)
    return dN_dxi, dN_deta

# Generate FEM-based displacement field
def generate_displacement_field_fem(deformation_mode='type0', max_displacement=6, randomness=0):
    """
    Generate displacement and strain fields using FEM-based deformation.

    Parameters:
    - deformation_mode: str, deformation mode type.
    - max_displacement: float, maximum displacement value (pixels).
    - randomness: float, controls randomness in certain deformation modes.

    Returns:
    - dx_total: np.ndarray, total displacement in x direction.
    - dy_total: np.ndarray, total displacement in y direction.
    - exx_total: np.ndarray, strain component exx.
    - eyy_total: np.ndarray, strain component eyy.
    - exy_total: np.ndarray, strain component exy.
    """
    # Generate mesh
    num_elem_x = 16  # Adjust as needed
    num_elem_y = 16
    node_coords, elements = generate_mesh(num_elem_x, num_elem_y, width, height)

    num_nodes = node_coords.shape[0]
    node_X = node_coords[:, 0]
    node_Y = node_coords[:, 1]

    # Center coordinates
    center_x, center_y = width / 2, height / 2

    # Select deformation mode
    if deformation_mode == 'stretching':
        # Stretching deformation
        epsilon_y = np.random.uniform(-0.1, 0.1)
        epsilon_x = 0.1 * np.sin(2 * np.pi * (node_Y - center_y) / height)
        node_dx = epsilon_x * (node_X - center_x)
        node_dy = epsilon_y * (node_Y - center_y)

    elif deformation_mode == 'rotation':
        # Rotation deformation
        angle = np.deg2rad(np.random.uniform(-8, 8))
        node_dx = (node_X - center_x) * (np.cos(angle) - 1) - (node_Y - center_y) * np.sin(angle)
        node_dy = (node_X - center_x) * np.sin(angle) + (node_Y - center_y) * (np.cos(angle) - 1)

    elif deformation_mode == 'translation':
        # Translation deformation
        delta_x = np.random.uniform(-10, 10)
        delta_y = np.random.uniform(-10, 10)
        node_dx = np.full_like(node_X, delta_x)
        node_dy = np.full_like(node_Y, delta_y)

    elif deformation_mode == 'shear':
        # Shear deformation
        shear_factor_x = np.random.uniform(-0.1, 0.1)
        shear_factor_y = np.random.uniform(-0.1, 0.1)
        node_dx = shear_factor_x * (node_Y - center_y)
        node_dy = shear_factor_y * (node_X - center_x)

    elif deformation_mode == 'scaling':
        # Scaling deformation
        scale_x = np.random.uniform(0.9, 1.1)
        scale_y = np.random.uniform(0.9, 1.1)
        node_dx = (scale_x - 1) * (node_X - center_x)
        node_dy = (scale_y - 1) * (node_Y - center_y)

    elif deformation_mode == 'nonlinear':
        # Nonlinear deformation, e.g., sine wave
        amplitude = np.random.uniform(0.2, 0.7)
        wavelength = np.random.uniform(30, 60)
        node_dx = amplitude * np.sin(2 * np.pi * node_Y / wavelength)
        node_dy = amplitude * np.sin(2 * np.pi * node_X / wavelength)

    elif deformation_mode == 'type0':
        # Type0 deformation
        random_factor_x = np.random.uniform(0, 1) * randomness
        random_factor_y = np.random.uniform(0, 1) * randomness
        node_dx = max_displacement * np.sin(np.pi * node_X / width + random_factor_x * np.pi) * \
                 np.cos(np.pi * node_Y / height + random_factor_x * np.pi)
        node_dy = max_displacement * np.cos(np.pi * node_X / width + random_factor_y * np.pi) * \
                 np.sin(np.pi * node_Y / height + random_factor_y * np.pi)

    elif deformation_mode == 'type1':
        # Type1 deformation
        random_factor_x = np.random.uniform(0, 1) * randomness
        random_factor_y = np.random.uniform(0, 1) * randomness
        node_dx = (max_displacement + random_factor_x) * np.sin(2 * np.pi * node_X / width + np.random.normal(0, 1) * np.pi) * \
                 np.cos(2 * np.pi * node_Y / height)
        node_dy = (max_displacement + random_factor_y) * np.cos(2 * np.pi * node_X / width + np.random.normal(0, 1) * np.pi) * \
                 np.sin(2 * np.pi * node_Y / height)

    elif deformation_mode == 'type2':
        # Type2 deformation
        random_factor_x = np.random.normal(0, 0.1, size=node_X.shape)
        random_factor_y = np.random.normal(0, 0.1, size=node_Y.shape)
        node_dx = (max_displacement + random_factor_x) * np.exp(-node_X / width) * \
                 np.sin(2 * np.pi * node_Y / height) + random_factor_x
        node_dy = (max_displacement + random_factor_y) * np.exp(-node_Y / height) * \
                 np.cos(2 * np.pi * node_X / width) + random_factor_y

    elif deformation_mode == 'type3':
        # Type3 deformation
        random_factor_x = np.random.normal(0, 0.5, size=node_X.shape)
        random_factor_y = np.random.normal(0, 0.5, size=node_Y.shape)
        node_dx = (np.random.normal(0, 9) + random_factor_x) * (node_X / width) ** 2
        node_dy = (np.random.normal(0, 9) + random_factor_y) * (node_Y / height) ** 2

    elif deformation_mode == 'type4':
        # Type4 deformation
        random_factor_x = np.random.normal(0, 1)
        random_factor_y = np.random.normal(0, 1)
        freq_x = np.random.normal(0, 3)
        freq_y = np.random.normal(0, 3)
        node_dx = (2 + random_factor_x) * np.sin(freq_x * np.pi * node_X / width) * \
                 np.cos(freq_y * np.pi * node_Y / height) + random_factor_x
        node_dy = (2 + random_factor_y) * np.cos(freq_x * np.pi * node_X / width) * \
                 np.sin(freq_y * np.pi * node_Y / height) + random_factor_y

    elif deformation_mode == 'custom':
        # Custom deformation
        max_strain = 0.2  # Maximum normal strain
        max_shear = 0.08  # Maximum shear strain
        random_coeff_F11 = np.random.uniform(-max_strain, max_strain)
        random_coeff_F12 = np.random.uniform(-max_shear, max_shear)
        random_coeff_F21 = np.random.uniform(-max_shear, max_shear)
        random_coeff_F22 = np.random.uniform(-max_strain, max_strain)
        F11 = 1 + random_coeff_F11 * (node_X - center_x) / width
        F12 = random_coeff_F12 * np.sin(2 * np.pi * (node_Y - center_y) / height)
        F21 = random_coeff_F21 * np.sin(2 * np.pi * (node_X - center_x) / width)
        F22 = 1 + random_coeff_F22 * (node_Y - center_y) / height
        node_dx = F11 * (node_X - center_x) + F12 * (node_Y - center_y) - (node_X - center_x)
        node_dy = F21 * (node_X - center_x) + F22 * (node_Y - center_y) - (node_Y - center_y)

    elif deformation_mode == 'sine_wave':
        # Sine wave deformation
        frequency_x = np.random.uniform(0, 30)  # Frequency in X direction
        frequency_y = np.random.uniform(0, 30)  # Frequency in Y direction
        node_dx = (max_displacement / 2) * np.sin(2 * np.pi * frequency_x * node_X / width)
        node_dy = (max_displacement / 2) * np.sin(2 * np.pi * frequency_y * node_Y / height)

    else:
        # Default zero displacement
        node_dx = np.zeros_like(node_X)
        node_dy = np.zeros_like(node_Y)

    node_displacements = np.column_stack((node_dx, node_dy))

    # Collect displacement and strain values
    x_coords = []
    y_coords = []
    dx_values = []
    dy_values = []
    exx_values = []
    eyy_values = []
    exy_values = []

    # Integration points in local coordinates
    num_integration_points = 3
    xi_vals = np.linspace(-1, 1, num_integration_points)
    eta_vals = np.linspace(-1, 1, num_integration_points)

    # Loop through each element
    for elem in elements:
        elem_node_indices = elem
        elem_node_coords = node_coords[elem_node_indices]
        elem_node_disp = node_displacements[elem_node_indices]

        # Loop through integration points
        for xi in xi_vals:
            for eta in eta_vals:
                N = shape_functions(xi, eta)
                dN_dxi, dN_deta = shape_function_derivatives(xi, eta)

                # Calculate Jacobian matrix
                J = np.zeros((2, 2))
                for i in range(9):
                    J[0, 0] += dN_dxi[i] * elem_node_coords[i, 0]
                    J[0, 1] += dN_dxi[i] * elem_node_coords[i, 1]
                    J[1, 0] += dN_deta[i] * elem_node_coords[i, 0]
                    J[1, 1] += dN_deta[i] * elem_node_coords[i, 1]
                detJ = np.linalg.det(J)
                if detJ == 0:
                    continue
                invJ = np.linalg.inv(J)

                # Calculate derivatives with respect to x and y
                dN_dx = np.zeros(9)
                dN_dy = np.zeros(9)
                for i in range(9):
                    dN_dx[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
                    dN_dy[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]

                # Calculate displacement at this point
                u = np.dot(N, elem_node_disp[:, 0])
                v = np.dot(N, elem_node_disp[:, 1])

                # Calculate strain at this point
                dudx = np.dot(dN_dx, elem_node_disp[:, 0])
                dudy = np.dot(dN_dy, elem_node_disp[:, 0])
                dvdx = np.dot(dN_dx, elem_node_disp[:, 1])
                dvdy = np.dot(dN_dy, elem_node_disp[:, 1])

                exx = dudx
                eyy = dvdy
                exy = 0.5 * (dudy + dvdx)

                # Map local coordinates to global coordinates
                x_global = np.dot(N, elem_node_coords[:, 0])
                y_global = np.dot(N, elem_node_coords[:, 1])

                x_coords.append(x_global)
                y_coords.append(y_global)
                dx_values.append(u)
                dy_values.append(v)
                exx_values.append(exx)
                eyy_values.append(eyy)
                exy_values.append(exy)

    # Interpolate to pixel grid
    points = np.column_stack((x_coords, y_coords))
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    dx_total_interp = griddata(points, dx_values, grid_points, method='linear', fill_value=0.0)
    dy_total_interp = griddata(points, dy_values, grid_points, method='linear', fill_value=0.0)
    exx_total_interp = griddata(points, exx_values, grid_points, method='linear', fill_value=0.0)
    eyy_total_interp = griddata(points, eyy_values, grid_points, method='linear', fill_value=0.0)
    exy_total_interp = griddata(points, exy_values, grid_points, method='linear', fill_value=0.0)

    dx_total = dx_total_interp.reshape((height, width))
    dy_total = dy_total_interp.reshape((height, width))
    exx_total = exx_total_interp.reshape((height, width))
    eyy_total = eyy_total_interp.reshape((height, width))
    exy_total = exy_total_interp.reshape((height, width))

    # Scale total displacement to maximum value
    dx_total *= max_displacement
    dy_total *= max_displacement

    # Calculate magnitude of the displacement field
    magnitude = np.sqrt(dx_total ** 2 + dy_total ** 2)
    max_magnitude = np.max(magnitude)

    # Normalize displacement field to a maximum value of 1
    if max_magnitude > 0:
        dx_total = dx_total / max_magnitude
        dy_total = dy_total / max_magnitude

    # Scale total displacement to maximum value
    dx_total *= max_displacement
    dy_total *= max_displacement

    # Return displacement and strain fields
    return dx_total, dy_total, exx_total, eyy_total, exy_total

# === Batch Processing Script ===

def batch_deform_images(
    input_dir,
    output_dir,
    num_steps=50,
    max_displacement=6,  # Total displacement set to 6 pixels
    randomness=0,
    crop_size=128,
    deformation_modes=None,
    global_start_count=85701  # Starting global counter
):
    """
    Apply various deformation modes to all BMP images in the input directory,
    decompose each deformation into 50 steps, apply 1/50 of the displacement amount at each step,
    and save the results.

    Parameters:
    - input_dir: str, path to the input directory containing original BMP images.
    - output_dir: str, path to the output directory to save results.
    - num_steps: int, number of incremental deformation steps (default is 50).
    - max_displacement: float, maximum total displacement value (pixels, default is 6).
    - randomness: float, controls randomness in certain deformation modes (default is 0).
    - crop_size: int, size of the central crop (crop_size x crop_size, default is 128).
    - deformation_modes: list of str, list of deformation modes (default is None, using predefined list).
    - global_start_count: int, starting value for the global counter (default is 85701).
    """
    if deformation_modes is None:
        deformation_modes = [
            'stretching', 'rotation', 'translation', 'shear', 'scaling', 'nonlinear',
            'type0', 'type1', 'type2', 'type3', 'type4', 'custom', 'sine_wave'
        ]

    # Define output subdirectories
    subdirs = {
        'images': 'deformed_images',
        'displacement': 'displacement_data',
        'strain': 'strain_data',
        'reference': 'reference_images'
    }

    # Create output subdirectories (if they do not exist)
    for key, subdir in subdirs.items():
        path = os.path.join(output_dir, subdir)
        os.makedirs(path, exist_ok=True)

    # Read input images
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.bmp')])
    input_images = []
    valid_image_files = []
    for file in image_files:
        image_path = os.path.join(input_dir, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and image.shape == (height, width):
            input_images.append(image)
            valid_image_files.append(file)
        else:
            print(f"Skipping file {file}: Not found or incorrect dimensions.")

    num_images = len(input_images)
    if num_images == 0:
        raise ValueError("No valid images found in the input directory.")

    print(f'Read {num_images} images from "{input_dir}".')

    # Initialize global deformation counter
    global_count = global_start_count

    # Generate pixel grid
    x = np.arange(0, width)
    y = np.arange(0, height)
    X, Y = np.meshgrid(x, y)

    # Loop through each image
    for img_idx, file in enumerate(tqdm(valid_image_files, desc='Processing images')):
        original_image = input_images[img_idx]
        print(f'Processing image: {file} ({img_idx + 1}/{num_images})')

        # Randomly select a deformation mode
        selected_deformation_mode = np.random.choice(deformation_modes)
        print(f'  Selected deformation mode: {selected_deformation_mode}')

        # Generate total displacement and strain fields
        dx_total, dy_total, exx_total, eyy_total, exy_total = generate_displacement_field_fem(
            deformation_mode=selected_deformation_mode,
            max_displacement=max_displacement,
            randomness=randomness
        )

        # Calculate displacement amount per step
        dx_step = dx_total / (num_steps)
        dy_step = dy_total / (num_steps)
        exx_step = exx_total / (num_steps)
        eyy_step = eyy_total / (num_steps)
        exy_step = exy_total / (num_steps)

        # Initialize cumulative displacement and strain fields
        dx_cumulative = np.zeros((height, width), dtype=np.float32)
        dy_cumulative = np.zeros((height, width), dtype=np.float32)
        exx_cumulative = np.zeros((height, width), dtype=np.float32)
        eyy_cumulative = np.zeros((height, width), dtype=np.float32)
        exy_cumulative = np.zeros((height, width), dtype=np.float32)

        # Apply incremental deformation
        for step in tqdm(range(1, num_steps + 1), desc=f'  Applying {num_steps} steps', leave=False):
            # Accumulate displacement and strain
            dx_cumulative += dx_step
            dy_cumulative += dy_step
            exx_cumulative += exx_step
            eyy_cumulative += eyy_step
            exy_cumulative += exy_step

            # Apply displacement to the image
            X_new = X + dx_cumulative
            Y_new = Y + dy_cumulative

            # Ensure coordinates are within image boundaries
            X_new_clipped = np.clip(X_new, 0, width - 1)
            Y_new_clipped = np.clip(Y_new, 0, height - 1)

            # Prepare coordinates for map_coordinates
            coordinates = np.vstack((Y_new_clipped.ravel(), X_new_clipped.ravel()))

            # Apply deformation
            deformed_image = map_coordinates(
                original_image, coordinates, order=1, mode='reflect'
            ).reshape((height, width))

            # Central crop
            start_x = (width - crop_size) // 2
            end_x = start_x + crop_size
            start_y = (height - crop_size) // 2
            end_y = start_y + crop_size

            deformed_image_cropped = deformed_image[start_y:end_y, start_x:end_x]
            original_image_cropped = original_image[start_y:end_y, start_x:end_x]
            dx_cropped = dx_cumulative[start_y:end_y, start_x:end_x]
            dy_cropped = dy_cumulative[start_y:end_y, start_x:end_x]
            exx_cropped = exx_cumulative[start_y:end_y, start_x:end_x]
            eyy_cropped = eyy_cumulative[start_y:end_y, start_x:end_x]
            exy_cropped = exy_cumulative[start_y:end_y, start_x:end_x]
            E_cropped = np.stack([exx_cropped, eyy_cropped, exy_cropped], axis=0)  # Shape (3, crop_size, crop_size)

            # Construct filenames
            deformation_image_name = f'deformation{global_count}_step{step}.bmp'
            deformation_image_path = os.path.join(output_dir, subdirs['images'], deformation_image_name)

            displacement_mat_name = f'displacement{global_count}_step{step}.mat'
            displacement_mat_path = os.path.join(output_dir, subdirs['displacement'], displacement_mat_name)

            strain_mat_name = f'strain{global_count}_step{step}.mat'
            strain_mat_path = os.path.join(output_dir, subdirs['strain'], strain_mat_name)

            reference_image_name = f'reference{global_count}_step{step}.bmp'
            reference_image_path = os.path.join(output_dir, subdirs['reference'], reference_image_name)

            # Save deformed image
            cv2.imwrite(deformation_image_path, deformed_image_cropped)

            # Save displacement data (using key 'uu', shape 2*128*128)
            uu = np.stack([dx_cropped, dy_cropped], axis=0)  # Shape (2, 128, 128)
            savemat(displacement_mat_path, {'uu': uu})

            # Save strain data (using key 'E', shape 3*128*128)
            savemat(strain_mat_path, {'E': E_cropped})

            # Save reference (original cropped) image
            cv2.imwrite(reference_image_path, original_image_cropped)

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f'    Completed step {step}/{num_steps}.')

        # Update global counter
        global_count += 1

        print(f'  Completed deformation for image "{file}".\n')

    print('Deformation processing completed for all images.')

# === Example Usage ===

if __name__ == '__main__':
    # Define input and output directories
    input_dir = r'E:\SwinT_UNET_data\large_deformation_dataset\original'
    output_dir = r'E:\SwinT_UNET_data\large_deformation_dataset'

    # Define deformation mode list
    deformation_modes = [
        'stretching', 'rotation', 'translation', 'shear', 'scaling', 'nonlinear',
        'type0', 'type1', 'type2', 'type3', 'type4', 'custom', 'sine_wave'
    ]

    # Call batch processing function
    batch_deform_images(
        input_dir=input_dir,
        output_dir=output_dir,
        num_steps=50,
        max_displacement=7,  # Set total displacement to 7 pixels
        randomness=0,
        crop_size=128,
        deformation_modes=deformation_modes,
        global_start_count=85701  # Adjust starting index as needed
    )