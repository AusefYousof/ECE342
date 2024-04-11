import time

import click
import cv2 as cv
import numpy as np
from serial import Serial
import torch
import sys
import io
import torchvision.transforms as transforms
import torch, torchvision, os, torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from PIL import Image
import pickle

PORT = "COM5"
BAUDRATE = 115200

PREAMBLE = "!START!\r\n"
DELTA_PREAMBLE = "!DELTA!\r\n"
SUFFIX = "!END!\r\n"

ROWS = 144
COLS = 174

import tkinter as tk
from tkinter import messagebox
def show_detection_alert():
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showinfo("Detection Alert", "PERSON DETECTED!")  # show an "Info" message box
    root.destroy()

class PD_CNN(nn.Module):
    def __init__(self):
        super(PD_CNN, self).__init__()
        # Increase convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adaptive Pooling to handle varying dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        # Revise fully connected layers according to new output size
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def load_model(model_path="PD_CNN"):
    model = torch.load(model_path)
    model.eval()  
    return model

def predict(model, img_bytes):
    """
    Make a prediction for an input image.
    """
    # Convert bytes back to PIL Image
    size = (174,144)
    img = Image.frombytes('L', size, img_bytes)

    #############################################
    #to save images (making dataset from scratch)
    ##############################################
    
    #save_dir = "1"
    #to save images (making dataset from scratch)
    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    #image_path = os.path.join(save_dir, f"image_{timestamp}.jpg")
    #img.save(image_path)

    #to output the image we received from stdin
    #img.show()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    # Apply transformations
    input_tensor = transform(img)


    #image_pil = to_pil_image(input_tensor)

    # Display the image and print its label
    #plt.imshow(image_pil, cmap='gray')
    #plt.show()
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    # Move to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
    
    # Convert output probabilities to predicted class (0 or 1)
    pred = torch.sigmoid(output).item() > 0.5
    return pred


CNNmodel = load_model()


@click.command()
@click.option(
    "-p", "--port", default=PORT, help="Serial (COM) port of the target board"
)
@click.option("-br", "--baudrate", default=BAUDRATE, help="Serial port baudrate")
@click.option("--timeout", default=1, help="Serial port timeout")
@click.option("--rows", default=ROWS, help="Number of rows in the image")
@click.option("--cols", default=COLS, help="Number of columns in the image")
@click.option("--preamble", default=PREAMBLE, help="Preamble string before the frame")
@click.option(
    "--delta_preamble",
    default=DELTA_PREAMBLE,
    help="Preamble before a delta update during video compression.",
)
@click.option(
    "--suffix", default=SUFFIX, help="Suffix string after receiving the frame"
)
@click.option(
    "--short-input",
    is_flag=True,
    default=False,
    help="If true, input is a stream of 4b values",
)
@click.option("--rle", is_flag=True, default=False, help="Run-Length Encoding")
@click.option("--quiet", is_flag=True, default=False, help="Print fewer messages")
def monitor(
    port: str,
    baudrate: int,
    timeout: int,
    rows: int,
    cols: int,
    preamble: str,
    delta_preamble: str,
    suffix: str,
    short_input: bool,
    rle: bool,
    quiet: bool,
) -> None:
    """
    Display images transferred through serial port. Press 'q' to close.
    """
    prev_frame_ts = None  # keep track of frames per second
    frame = None

    click.echo(f"Opening communication on port {port} with baudrate {baudrate}")

    if isinstance(suffix, str):
        suffix = suffix.encode("ascii")

    if isinstance(preamble, str):
        preamble = preamble.encode("ascii")

    if isinstance(delta_preamble, str):
        delta_preamble = delta_preamble.encode("ascii")

    img_rx_size = rows * cols
    if short_input:
        img_rx_size //= 2
    if rle:
        img_rx_size *= 2

    partial_frame_counter = 0  # count partial updates every full frame

    with Serial(port, baudrate, timeout=timeout) as ser:
        while True:
            if not quiet:
                click.echo("Waiting for input data...")

            full_update = wait_for_preamble(ser, preamble, delta_preamble)

            if full_update:
                click.echo(
                    f"Full update (after {partial_frame_counter} partial updates)"
                )
                partial_frame_counter = 0
            else:
                if not quiet:
                    click.echo("Partial update")
                partial_frame_counter += 1

                if frame is None:
                    click.echo(
                        "No full frame has been received yet. Skipping partial update."
                    )
                    continue

            if not quiet:
                click.echo("Receiving picture...")

            try:
                raw_data = get_raw_data(ser, img_rx_size, suffix)
                if not quiet:
                    click.echo(f"Received {len(raw_data)} bytes")
            except ValueError as e:
                click.echo(f"Error while waiting for frame data: {e}")

            if short_input:
                raw_data = (
                    expand_4b_to_8b(raw_data)
                    if not rle
                    else expand_4b_to_8b_rle(raw_data)
                )
            elif rle and len(raw_data) % 2 != 0:
                # sometimes there serial port picks up leading 0s
                # discard these
                raw_data = raw_data[1:]

            if rle:
                raw_data = decode_rle(raw_data)

            try:
                new_frame = load_raw_frame(raw_data, rows, cols)
            except ValueError as e:
                click.echo(f"Malformed frame. {e}")
                continue

            frame = new_frame if full_update else frame + new_frame

            now = time.time()
            if prev_frame_ts is not None:
                try:
                    fps = 1 / (now - prev_frame_ts)
                    click.echo(f"Frames per second: {fps:.2f}")
                except ZeroDivisionError:
                    click.echo("FPS too fast to measure")
            prev_frame_ts = now


            if frame is not None:
                
                
                frame_bytes = pickle.dumps(frame)
                prediction = predict(CNNmodel, frame_bytes)
                print(f"Prediction: {prediction}")
                if prediction:
                    show_detection_alert()


            cv.namedWindow("Video Stream", cv.WINDOW_KEEPRATIO)
            cv.imshow("Video Stream", frame)

            # Wait for 'q' to stop the program
            if cv.waitKey(1) == ord("q"):
                break

    cv.destroyAllWindows()


def wait_for_preamble(ser: Serial, preamble: str, partial_preamble: str) -> bool:
    """
    Wait for a preamble string in the serial port.

    Returns `True` if next frame is full, `False` if it's a partial update.
    """
    while True:
        try:
            line = ser.readline()
            if line == preamble:
                return True
            elif line == partial_preamble:
                return False
        except UnicodeDecodeError:
            pass


def get_raw_data(ser: Serial, num_bytes: int, suffix: bytes = b"") -> bytes:
    """
    Get raw frame data from the serial port.
    """
    rx_max_len = num_bytes + len(suffix)
    max_tries = 10_000
    raw_img = b""

    for _ in range(max_tries):
        raw_img += ser.read(max(1, ser.in_waiting))

        suffix_idx = raw_img.find(suffix)
        if suffix_idx != -1:
            raw_img = raw_img[:suffix_idx]
            break

        if len(raw_img) >= rx_max_len:
            raw_img = raw_img[:num_bytes]
            break
    else:
        raise ValueError("Max tries exceeded.")

    return raw_img


def expand_4b_to_8b(raw_data: bytes) -> bytes:
    """
    Transform an input of 4-bit encoded values into a string of 8-bit values.

    For example, value 0xFA gets converted to [0xF0, 0xA0]
    """
    return bytes(val for pix in raw_data for val in [pix & 0xF0, (pix & 0x0F) << 4])


def expand_4b_to_8b_rle(raw_data: bytes) -> bytes:
    """
    Transform an input of 4-bit encoded RLE values into a string of 8-bit values.

    For example, value 0xFA gets converted to [0xF0, 0x0A]
    """
    return bytes(val for pix in raw_data for val in [pix & 0xF0, pix & 0x0F])


def decode_rle(raw_data: bytes) -> bytes:
    """
    Decode Run-Length Encoded data.
    """
    return bytes(
        val
        for pix, count in zip(raw_data[::2], raw_data[1::2])
        for val in [pix] * count
    )


def load_raw_frame(raw_data: bytes, rows: int, cols: int) -> np.array:
    return np.frombuffer(raw_data, dtype=np.uint8).reshape((rows, cols, 1))



if __name__ == "__main__":
    monitor()

