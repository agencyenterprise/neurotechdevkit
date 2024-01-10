<template>
  <fieldset>
    <div class="row">
      <div class="col mb-3">
        <label for="positionX" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The coordinate (in meters) of the point at the center of the transducer">Position
          X</label>
        <input type="number" step="any" class="form-control" name="positionX" placeholder="0.0" required />
      </div>
      <div class="col mb-3">
        <label for="positionY" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The coordinate (in meters) of the point at the center of the transducer">Position
          Y</label>
        <input type="number" step="any" class="form-control" name="positionY" placeholder="0.0" required />
      </div>
      <div class="col mb-3" data-3d-only="true" hidden>
        <label for="positionZ" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The coordinate (in meters) of the point at the center of the transducer">Position
          Z</label>
        <input type="number" step="any" class="form-control" name="positionZ" placeholder="0.0"
          data-3d-input-only="true" />
      </div>
    </div>
    <div class="row">
      <div class="col mb-3">
        <label for="directionX" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Indicate the direction the source is pointing">Direction X</label>
        <input type="number" step="any" class="form-control" name="directionX" placeholder="0.0" required />
      </div>
      <div class="col mb-3">
        <label for="directionY" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Indicate the direction the source is pointing">Direction Y</label>
        <input type="number" step="any" class="form-control" name="directionY" placeholder="1.0" required />
      </div>
      <div class="col mb-3" data-3d-only="true" hidden>
        <label for="directionZ" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Indicate the direction the source is pointing">Direction Z</label>
        <input type="number" step="any" class="form-control" name="directionZ" placeholder="0.0"
          data-3d-input-only="true" />
      </div>
    </div>
    <div class="mb-3">
      <label for="aperture" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The aperture (in meters) of the transducer">Aperture</label>
      <input type="number" step="any" class="form-control" name="aperture" placeholder="0.064" required />
    </div>
    <div class="mb-3">
      <label for="focalLength" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The focal length (in meters) of the transducer">Focal Length</label>
      <input type="number" step="any" class="form-control" name="focalLength" placeholder="0.064" required />
    </div>
    <div class="mb-3">
      <label for="numPoints" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The number of point sources to use when simulating the transducer">Points</label>
      <input type="number" class="form-control" name="numPoints" placeholder="20000" required />
    </div>
    <div class="mb-3">
      <label for="delay" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The delay (in seconds) that the source will wait before emitting">Delay</label>
      <input type="number" step="any" class="form-control" name="delay" placeholder="0.0" required />
    </div>
  </fieldset>
</template>

<script>
export default {

}
</script>

<style scoped></style>
