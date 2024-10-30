"""
Web app for `palettecleanse`
"""

from pathlib import Path

import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request, send_from_directory
from palettecleanse.palette import Palette

app = Flask(__name__)

UPLOAD_FOLDER = Path("tmp")
UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file part", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    n_colors = request.form.get("integerInput", type=int)

    filename = file.filename
    original_path = UPLOAD_FOLDER / filename
    file.save(original_path)

    # generate & save images
    im_pal = Palette(original_path, n_colors=n_colors)
    im_pal.display_all_palettes()
    all_palettes = plt.gcf()
    palette_path = UPLOAD_FOLDER / f"all_palettes_{filename}"
    all_palettes.savefig(palette_path)

    im_pal.display_example_plots()
    example_plots = plt.gcf()
    plots_path = UPLOAD_FOLDER / f"example_plots_{filename}"
    example_plots.savefig(plots_path)

    rgb_values = str(im_pal.rgb_values)
    hex_values = im_pal.hex_values

    return render_template(
        "result.html",
        original_image=filename,
        processed_images=[f"all_palettes_{filename}", f"example_plots_{filename}"],
        list_one=rgb_values,
        list_two=hex_values,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/cleanup", methods=["POST"])
def cleanup():
    data = request.get_json()
    original_image = data.get("original_image")
    processed_images = data.get("processed_images", [])

    response_messages = []

    # delete images once processed
    if original_image:
        original_path = UPLOAD_FOLDER / original_image
        if original_path.exists():
            original_path.unlink()
            response_messages.append(f"Deleted: {original_image}")

    for processed_image in processed_images:
        processed_path = UPLOAD_FOLDER / processed_image
        if processed_path.exists():
            processed_path.unlink()
            response_messages.append(f"Deleted: {processed_image}")

    return jsonify({"message": response_messages}), 200


if __name__ == "__main__":
    app.run(debug=True)
