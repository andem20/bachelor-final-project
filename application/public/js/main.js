const red = [230, 69, 67]
const green = [38, 209, 83]

const classColors = {
    Malignant: red,
    Benign: green
}

async function preprocessImage(file) {
    const fileType = file.type.split("/")[1]

    const response = await fetch(`/preprocess/${fileType}`, {
        method: "POST",
        body: file
    })

    const image_base64 = await response.text()

    const image = new Image()
    image.src = "data:image/png;base64," + image_base64

    return image
}

function createCanvasLayer(zIndex, width, height) {
    const canvas = document.createElement("canvas")
    canvas.className = "image_preview"
    canvas.width = width
    canvas.height = height

    return [canvas, canvas.getContext('2d')]
}

async function analyseImage(imageDataBase64, roiConfidenceThreshold) {
    const response = await fetch(`/mammogram/analysis/${roiConfidenceThreshold}`, {
        method: "POST",
        body: imageDataBase64.split("base64,")[1]
    });

    return await response.json()
}

function createSegmentationBuffer(width, height, segmentation, color) {
    const buffer = new Uint8ClampedArray(width * height * 4);
    
    for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4;
            const pixelValue = segmentation[y * height + x]
            const pixelValueNorm = pixelValue / 0xff
            buffer[pos] = color[0] * pixelValueNorm
            buffer[pos + 1] = color[1] * pixelValueNorm
            buffer[pos + 2] = color[2] * pixelValueNorm
            buffer[pos + 3] = pixelValue / 2
        }
    }

    return buffer
}

function drawSegmentation(x, y, width, height, ctx, segmentation, color) {
    const buffer = createSegmentationBuffer(
        width,
        height,
        segmentation,
        color
    )

    const imageData = ctx.createImageData(width, height)
    imageData.data.set(buffer)
    ctx.putImageData(imageData, x, y);
}

function drawBoundingBox(x, y, width, height, ctx, color, textLineArray) {
    ctx.beginPath();
    ctx.lineWidth = 10;
    ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
    ctx.rect(x, y, width, height);
    ctx.stroke();

    ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
    ctx.beginPath()
    ctx.fillRect(x - (ctx.lineWidth / 2), y - 120, 500, 120)

    ctx.fillStyle = "black"
    ctx.font = "50px arial"

    for(let i = 0; i < textLineArray.length; i++) {
        ctx.fillText(textLineArray[i], x, y + (i*50) - 70)
    }
}

function createSpinner() {
    const spinner = document.createElement("div")
    spinner.id = "spinner"

    const container = document.createElement("div")
    container.className = "lds-roller"

    for (let i = 0; i < 8; i++) {
        container.appendChild(document.createElement("div"))
    }
    const status = document.createElement("div")

    spinner.appendChild(container)
    spinner.appendChild(status)

    return [spinner, status]
}


async function handleImageUpload(file, previewContainer, confidenceSelector) {
    previewContainer.innerHTML = ""
    const [spinner, status] = createSpinner()

    previewContainer.appendChild(spinner)

    spinner.style.display = "block"
    status.innerText = "Loading image..."

    const image = await preprocessImage(file)

    image.onload = async function () {
        status.innerText = "Analyzing..."

        const [originalCanvas, originalCtx] = createCanvasLayer(1, this.width, this.height)

        previewContainer.appendChild(originalCanvas)

        originalCtx.drawImage(image, 0, 0);
        originalCanvas.style.opacity = 0.2
        originalCanvas.style.height = 100 + "%"

        const imageDataBase64 = originalCanvas.toDataURL("image/png");
        const json = await analyseImage(imageDataBase64, confidenceSelector.value)

        for (let i = 0; i < json["segmentations"].length; i++) {
            const [overlayCanvas, overlayCtx] = createCanvasLayer(2 + i, this.width, this.height)
            previewContainer.appendChild(overlayCanvas)

            const { segmentation, width, height, roi_bbox, bbox, bboxConfidence, pathology } = json["segmentations"][i]
            const pathologyName = pathology > 0.5 ? "Malignant" : "Benign"
            const color = classColors[pathologyName]

            drawSegmentation(roi_bbox[1], roi_bbox[0], width, height, overlayCtx, segmentation, color)

            const percentage = pathology > 0.5 ? pathology : 1 - pathology
            const textLineArray = [`${(percentage * 100).toFixed(1)}% ${pathologyName}`, `${(bboxConfidence * 100).toFixed(1)}% roi confidence`]

            const widthMargin = width * 0.1
            const heightMargin = height * 0.1

            drawBoundingBox(
                Math.max(0, bbox[1] - widthMargin), 
                Math.max(0, bbox[0] - heightMargin), 
                Math.min(bbox[2] + widthMargin * 2, this.width), 
                Math.min(bbox[3] + heightMargin * 2, this.height), 
                overlayCtx, 
                color, 
                textLineArray
            )
            overlayCanvas.style.opacity = 1
            overlayCanvas.style.height = 100 + "%"
        }

        originalCanvas.style.opacity = 1
        spinner.style.display = "none"
    }
}