function BtnLoading(elem) {
  $(elem).attr("data-original-text", $(elem).html())
  $(elem).prop("disabled", true)
  $(elem).html('<i class="spinner-border spinner-border-sm"></i>  ...')
}

function BtnReset(elem) {
  $(elem).prop("disabled", false)
  $(elem).html($(elem).attr("data-original-text"))
}

$(document).ready(function () {
  $('[data-bs-toggle="tooltip"]').tooltip()
})


function plotToCanvas({xlim, ylim, xlabel, ylabel, xticks, yticks, image, clickCallback}){
    // Define the size of the ticks
    const tickSize = 10;
    const leftMargin = 75;
    const bottomMargin = 65;
    const topMargin = 30;
    const rightMargin = 30;
    const xLabelMargin = 30
    const yLabelMargin = topMargin - 20

    // Create a new canvas and get the context
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';

    const img = new Image();

    img.onload = function() {
        // Set the canvas dimensions
        canvas.width = img.width + leftMargin + rightMargin;
        canvas.height = img.height + bottomMargin + topMargin;

        // Draw the image on the canvas
        ctx.drawImage(img, leftMargin, topMargin);

        // // Save the data ranges
        canvas.dataset.xlim = JSON.stringify(xlim);
        canvas.dataset.ylim = JSON.stringify(ylim);

        ctx.font = '24px DejaVu Sans';
        // Draw the x-axis label
        ctx.fillText(xlabel, canvas.width / 2, canvas.height - 5);

        // Draw the y-axis label
        // Rotation is needed to draw vertical text
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-90 * Math.PI / 180);
        ctx.fillText(ylabel, 0, 0);
        ctx.restore();

        ctx.font = '14px DejaVu Sans';
        for (const tick of xticks) {
            const x = leftMargin + ((tick - xlim[0]) / (xlim[1] - xlim[0])) * img.width;

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
            ctx.translate(x, (canvas.height - topMargin + tickSize) - yLabelMargin);

            // Rotate the context
            ctx.rotate(-45 * Math.PI / 180);

            // Render the text, adjusting the x position due to rotation
            ctx.fillText(tick.toFixed(3), 0, 0);

            // Restore the context state
            ctx.restore();
        }
        for (const tick of yticks) {
            const y = img.height - ((tick - ylim[0]) / (ylim[1] - ylim[0])) * img.height;
            // Draw the tick
            ctx.beginPath();
            ctx.moveTo(leftMargin - tickSize, y + topMargin);
            ctx.lineTo(leftMargin, y + topMargin);
            ctx.stroke();

            const textCenterMargin = 5; // to make the text centered
            ctx.fillText(tick.toFixed(3), xLabelMargin, y + topMargin + textCenterMargin);
        }
    };

    // Set the image source
    img.src = 'data:image/png;base64,' + image;


    function getMouseCoordinates(canvas, event) {
        // Get the size of the canvas
        const rect = canvas.getBoundingClientRect();

        // Calculate the position of the click in pixels
        let xPixel = (event.clientX - rect.left);
        let yPixel = (event.clientY - rect.top)
        if (xPixel < leftMargin || yPixel < topMargin) return null
        if (xPixel > img.width + leftMargin || yPixel > img.height + topMargin) return null
        xPixel -= leftMargin;
        yPixel -= topMargin;

        // Get the data ranges
        const xlim = JSON.parse(canvas.dataset.xlim);
        const ylim = JSON.parse(canvas.dataset.ylim);

        // Convert from pixels to data units
        const xData = xlim[0] + (xlim[1] - xlim[0]) * xPixel / img.width;
        const yData = ylim[1] - (ylim[1] - ylim[0]) * yPixel / img.height;
        return [xData, yData]
    }

    // Handle click events on the canvas
    canvas.addEventListener('click', function(event) {
      const mouseCoordinates = getMouseCoordinates(canvas, event)
      if (mouseCoordinates == null) return
      const [xData, yData] = mouseCoordinates
      clickCallback(xData, yData)
    });

    canvas.addEventListener('mousemove', function(event) {
      // Create the longest possible text
      let text = `${xlabel}: -99.999, ${ylabel}: -99.999`;
      const textMeasure = ctx.measureText(text);
    
      // Calculate text width to position the rectangle and text
      const rectWidth = textMeasure.width + 30; // add some padding
      const rectHeight = 30; // roughly two lines of text
      const rectX = (leftMargin + (img.width/2)) - (rectWidth/2);
      const rectY = 0; // top of the canvas

      // Clear a rectangle in the top right corner for the cursor position
      ctx.clearRect(rectX, rectY, rectWidth, rectHeight);

      // Draw rectangle
      ctx.fillStyle = "white";
      ctx.fillRect(rectX, rectY, rectWidth, rectHeight);

      const mouseCoordinates = getMouseCoordinates(canvas, event)
      if (mouseCoordinates == null) return
      const [dataX, dataY] = mouseCoordinates
      text = `${xlabel}: ${dataX.toFixed(3)}, ${ylabel}: ${dataY.toFixed(3)}`;

      // Canvas text styles
      ctx.fillStyle = 'black';
      ctx.font = '16px Arial';

      // Display the text within the rectangle
      const textX = rectX + rectWidth/2 - textMeasure.width/2;
      const textY = 20;  // roughly one line down from the top edge

      ctx.fillText(text, textX, textY);

  });
    return canvas
}