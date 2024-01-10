<template>
  <form id="displayForm">
    <fieldset>
      <div class="mb-3">
        <label for="sliceAxis" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Axis along which to slice the 3D field to be recorded">Axis</label>
        <select class="form-select" aria-label="Axis" id="sliceAxis" name="sliceAxis" onchange="valueChanged()"
          data-3d-input-only="true">
          <option value="" selected>Select an axis</option>
          <option value="x">X</option>
          <option value="y">Y</option>
          <option value="z">Z</option>
        </select>
      </div>
      <div class="mb-3">
        <label for="slicePosition" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The position (in meters) along the slice axis at which the slice of the 3D field should be made">Distance
          from Origin (m)</label>
        <input type="number" step="any" class="form-control" id="slicePosition" name="slicePosition" placeholder="0.0"
          onchange="valueChanged()" data-3d-input-only="true" />
      </div>
    </fieldset>
  </form>
</template>

<script>
export default {
}
</script>

<style scoped></style>
