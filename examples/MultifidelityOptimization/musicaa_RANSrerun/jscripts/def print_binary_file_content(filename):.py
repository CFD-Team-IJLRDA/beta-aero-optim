import numpy as np

def parse_binary_file(filename):
    residuals = []

    with open(filename, 'rb') as f:
        while True:
            try:
                # Read the first marker (assuming 4 bytes)
                marker1 = np.fromfile(f, dtype='>i4', count=1)[0]

                # Read the residual values (assuming 6 floating-point numbers, each 8 bytes)
                residual_values = {
                    'var1': np.fromfile(f, dtype='>f8', count=1)[0],
                    'var2': np.fromfile(f, dtype='>f8', count=1)[0],
                    'var3': np.fromfile(f, dtype='>f8', count=1)[0],
                    'var4': np.fromfile(f, dtype='>f8', count=1)[0],
                    'var5': np.fromfile(f, dtype='>f8', count=1)[0],
                    'var6': np.fromfile(f, dtype='>f8', count=1)[0]
                }

                # Read the second marker (assuming 4 bytes)
                marker2 = np.fromfile(f, dtype='>i4', count=1)[0]

                # Append the extracted residuals to the list
                residuals.append(residual_values)

            except IndexError:
                # Break the loop if the end of the file is reached
                break

    return residuals


# Example usage
residuals = parse_binary_file('/home/johnson/Desktop/MUSICAA/OGV_RANS/OGV_RANS_20241007_2/residuals.bin')

# Print the extracted residuals
for i, residual in enumerate(residuals):
    print(f"Record {i+1}: {residual}")
