function BtnLoading(elem) {
  $(elem).attr("data-original-text", $(elem).html());
  $(elem).prop("disabled", true);
  $(elem).html('<i class="spinner-border spinner-border-sm"></i>  ...');
}

function BtnReset(elem) {
  $(elem).prop("disabled", false);
  $(elem).html($(elem).attr("data-original-text"));
}

$(document).ready(function () {
  $('[data-bs-toggle="tooltip"]').tooltip();
});

function plotToCanvas({
  xlim,
  ylim,
  xlabel,
  ylabel,
  xticks,
  yticks,
  image,
  clickCallback,
  drawLine,
}) {
  // Renders a matplotlib plot to a canvas element

  const tickSize = 10;
  const leftMargin = 80;
  const bottomMargin = 65;
  const topMargin = 30;
  const rightMargin = 30;
  const xLabelMargin = leftMargin - 50;
  const yLabelMargin = topMargin - 25;

  // Variables for drawing a line
  // Variables to track line start and end position
  let start = null;
  let end = null;

  // Create a canvas and get the context
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";

  const img = new Image();

  img.onload = function () {
    // Set the canvas dimensions
    canvas.width = img.width + leftMargin + rightMargin;
    canvas.height = img.height + bottomMargin + topMargin;

    drawPlot();
  };

  // Set the image source
  img.src = "data:image/png;base64," + image;

  function drawPlot() {
    // Draw everything to the canvas
    ctx.drawImage(img, leftMargin, topMargin);

    // // Save the data ranges
    canvas.dataset.xlim = JSON.stringify(xlim);
    canvas.dataset.ylim = JSON.stringify(ylim);

    ctx.font = "20px manropebold";
    ctx.fillStyle = "#4F4F4F";

    // Draw the x-axis label
    ctx.fillText(xlabel, canvas.width / 2, canvas.height - 5);

    // Draw the y-axis label
    // Rotation is needed to draw vertical text
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate((-90 * Math.PI) / 180);
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();

    ctx.font = "13px manropebold";

    // Draw the x-axis ticks and labels
    for (const tick of xticks) {
      const x =
        leftMargin + ((tick - xlim[0]) / (xlim[1] - xlim[0])) * img.width;

      if (x < leftMargin || x > img.width + leftMargin) {
        continue;
      }

      // Draw the tick
      ctx.beginPath();
      ctx.moveTo(x, canvas.height - bottomMargin);
      ctx.lineTo(x, canvas.height - bottomMargin + tickSize);
      ctx.stroke();

      // Draw the label with rotation
      // Save the current context state
      ctx.save();

      // Translate to the desired point of rotation
      ctx.translate(x, canvas.height - topMargin + tickSize - yLabelMargin);

      // Rotate the context
      ctx.rotate((-45 * Math.PI) / 180);

      // Render the text, adjusting the x position due to rotation
      ctx.fillText(tick.toFixed(3), 0, 0);

      // Restore the context state
      ctx.restore();
    }

    // Draw the y-axis ticks and labels
    for (const tick of yticks) {
      const y =
        topMargin +
        img.height -
        ((tick - ylim[0]) / (ylim[1] - ylim[0])) * img.height;
      if (y > img.height) {
        continue;
      }

      // Draw the tick
      ctx.beginPath();
      ctx.moveTo(leftMargin - tickSize, y);
      ctx.lineTo(leftMargin, y);
      ctx.stroke();

      const textCenterMargin = 5; // to make the text centered
      ctx.fillText(tick.toFixed(3), xLabelMargin, y + textCenterMargin);
    }
  }

  function plotCoordinates(event) {
    // Create the longest text we will need to display
    let text = `${xlabel}: -99.999, ${ylabel}: -99.999`;
    const textMeasure = ctx.measureText(text);

    // Calculate text width to position the rectangle and text
    const rectWidth = textMeasure.width + 30; // add some padding
    const rectHeight = 30;
    const rectX = leftMargin + img.width / 2 - rectWidth / 2;
    const rectY = 0; // top of the canvas

    // Clear a rectangle for the cursor position
    ctx.clearRect(rectX, rectY, rectWidth, rectHeight);

    // Draw rectangle
    ctx.fillStyle = "white";
    ctx.fillRect(rectX, rectY, rectWidth, rectHeight);

    const mouseCoordinates = getMouseCoordinates(event.clientX, event.clientY);
    if (mouseCoordinates == null) return; // outside the image
    const [dataX, dataY] = mouseCoordinates;
    text = `${xlabel}: ${dataX.toFixed(3)}, ${ylabel}: ${dataY.toFixed(3)}`;

    // Canvas text styles
    ctx.fillStyle = "#4F4F4F";
    ctx.font = "16px manropebold";

    // Display the text within the rectangle
    const textX = rectX + rectWidth / 2 - textMeasure.width / 2;
    const textY = 20;

    ctx.fillText(text, textX, textY);
  }

  function getMouseCoordinates(clientX, clientY) {
    // convert mouse coordinates in pixels to data coordinates

    // Get the size of the canvas
    const rect = canvas.getBoundingClientRect();

    // Calculate the position of the click in pixels
    let xPixel = clientX - rect.left;
    let yPixel = clientY - rect.top;
    if (xPixel < leftMargin || yPixel < topMargin) return null;
    if (xPixel > img.width + leftMargin || yPixel > img.height + topMargin)
      return null;
    xPixel -= leftMargin;
    yPixel -= topMargin;

    // Get the data ranges
    const xlim = JSON.parse(canvas.dataset.xlim);
    const ylim = JSON.parse(canvas.dataset.ylim);

    // Convert from pixels to data units
    const xData = xlim[0] + ((xlim[1] - xlim[0]) * xPixel) / img.width;
    const yData = ylim[1] - ((ylim[1] - ylim[0]) * yPixel) / img.height;
    return [xData, yData];
  }

  canvas.addEventListener("click", function (e) {
    // Handler for when we want to draw a line
    if (!drawLine) return;

    // Check if the click is outside the image
    if (getMouseCoordinates(e.clientX, e.clientY) == null) return;

    // Get the current mouse position
    const rect = this.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (!start) {
      // If there is no start position, this click will define the start of the line
      start = { x, y };
    } else {
      // If there is a start position, this click will define the end of the line
      end = { x, y };

      // Draw the line from the starting position to the ending position
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();

      const endCoordinates = getMouseCoordinates(e.clientX, e.clientY);
      if (endCoordinates == null) return;
      const [endX, endY] = endCoordinates;

      const startCoordinates = getMouseCoordinates(
        start.x + rect.left,
        start.y + rect.top,
      );
      if (startCoordinates == null) return;
      const [startX, startY] = startCoordinates;
      if (clickCallback) {
        clickCallback(startX, startY, endX, endY);
      } else {
        console.log(`start: ${startX}, ${startY}, end: ${endX}, ${endY}`);
        console.log("No callback function defined");
      }

      // Reset the start and end positions
      start = null;
      end = null;
    }
  });

  canvas.addEventListener("click", function (event) {
    // Handler for when we don't want to draw a line, only get the coordinates of the click
    if (!drawLine) {
      if (!clickCallback) {
        console.log("No callback function defined");
        return;
      }
      const mouseCoordinates = getMouseCoordinates(
        event.clientX,
        event.clientY,
      );
      if (mouseCoordinates == null) return;
      const [xData, yData] = mouseCoordinates;
      clickCallback(xData, yData);
    }
  });

  canvas.addEventListener("mousemove", function (e) {
    // Handler for when we want to draw a line
    if (drawLine) {
      // If there is a start position but no end position, draw a line from the start position to the current mouse position.
      if (start && !end) {
        // Get the current mouse position
        const rect = this.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if the mouse is outside the image
        if (x < leftMargin || y < topMargin) return;
        if (x > img.width + leftMargin || y > img.height + topMargin) return;

        // clear the canvas to get rid of the previous line preview
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the plot again
        drawPlot();

        // Calculate the angle between the x-axis and the line from start to cursor position
        const angle = Math.atan2(y - start.y, x - start.x);

        // Define the lengths of the additional lines
        const length = 50;

        // Draw lines at 90 degrees and 45 degrees from the main line
        const offsetAngles = [Math.PI / 2, Math.PI / 4]; // 90 and 45 degrees
        for (let offset of offsetAngles) {
          for (let direction of [-1, 1]) {
            // Clockwise and counter-clockwise
            const lineX =
              start.x + length * Math.cos(angle + direction * offset);
            const lineY =
              start.y + length * Math.sin(angle + direction * offset);

            // Draw the line
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(lineX, lineY);
            ctx.strokeStyle = "red"; // Set color to red for the additional lines
            ctx.stroke();
          }
        }
        ctx.strokeStyle = "black"; // Set color back to black for the main line

        // Draw the live preview of the main line
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
    plotCoordinates(e);
  });

  return canvas;
}
