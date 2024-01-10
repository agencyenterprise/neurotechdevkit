<template>
  <fieldset>
    <div class="mb-3">
      <div class="btn-group">
        <input type="radio" class="btn-check" name="options" id="isPreBuilt" autocomplete="off"
          onchange="togglePreBuiltDiv()" checked />
        <label class="btn btn-primary" for="isPreBuilt" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Use a prebuilt scenario with a target and transducers">Prebuilt</label>

        <input type="radio" class="btn-check" name="options" id="isCtScan" autocomplete="off"
          onchange="toggleCTFileDiv()" />
        <label class="btn btn-primary" for="isCtScan" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Load a CT scan from a file">CT Scan</label>
      </div>
    </div>

    <div class="mb-3">
      <div id="preBuiltDiv">
        <label for="scenario" class="form-label">Scenarios:</label>
        <select name="scenario" id="scenario" class="form-select" onchange="scenarioSelected(this)"></select>
      </div>
      <div id="ctFileDiv" hidden>
        <form id="ctForm">
          <div class="mb-3">
            <label class="form-label" for="loadedCTs">Available CTs</label>
            <select class="form-select" size="3" name="loadedCTs" id="loadedCTs" onchange="ctSelected(this)" required>
            </select>
          </div>
          <div class="mb-3" data-2d-only="true">
            <label for="ctSliceAxis" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
              title="Axis along which to slice the 3D field to be recorded">Axis</label>
            <select class="form-select" aria-label="Axis" id="ctSliceAxis" data-2d-input-only="true" name="ctSliceAxis"
              required onchange="ctSliceAxisChanged(this)">
              <option value selected>Select an axis</option>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z">Z</option>
            </select>
          </div>
          <div class="mb-3" data-2d-only="true">
            <label for="ctSlicePosition" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
              title="The position (in meters) along the slice axis at which the slice of the 3D field should be made">Distance
              from Origin (m)</label>
            <input type="number" step="any" class="form-control" id="ctSlicePosition" name="ctSlicePosition"
              data-2d-input-only="true" placeholder="0.0" required onchange="valueChanged()" />
          </div>
        </form>
        <div class="mb-3">
          <label for="ctFiles" class="form-label"
            title="Select the CT file and the file containing the mapping between layers and masks">CT
            and mapping
            Files</label>
          <input class="form-control" type="file" id="ctFiles" onchange="filesChosen(this)" multiple />
        </div>
        <div class="mb-3">
          <button id="fileUploadButton" class="btn btn-primary" type="button" onclick="fileUploadClicked(this)" disabled>
            Upload new CT
          </button>
        </div>
      </div>
    </div>
  </fieldset>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>

<style scoped></style>
