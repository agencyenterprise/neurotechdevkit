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
        <input type="number" step="any" class="form-control" name="directionX" placeholder="1.0" required />
      </div>
      <div class="col mb-3">
        <label for="directionY" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Indicate the direction the source is pointing">Direction Y</label>
        <input type="number" step="any" class="form-control" name="directionY" placeholder="0.0" required />
      </div>
      <div class="col mb-3" data-3d-only="true" hidden>
        <label for="directionZ" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Indicate the direction the source is pointing">Direction Z</label>
        <input type="number" step="any" class="form-control" name="directionZ" placeholder="0.0"
          data-3d-input-only="true" />
      </div>
    </div>
    <div class="row" data-3d-only="true" hidden>
      <div class="col mb-3">
        <label for="centerLineX" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The input value for the direction the center of elements should have">Center
          Line X</label>
        <input type="number" step="any" class="form-control" name="centerLineX" placeholder="0.0"
          data-3d-input-only="true" />
      </div>
      <div class="col mb-3">
        <label for="centerLineY" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The input value for the direction the center of elements should have">Center
          Line Y</label>
        <input type="number" step="any" class="form-control" name="centerLineY" placeholder="0.0"
          data-3d-input-only="true" />
      </div>
      <div class="col mb-3">
        <label for="centerLineZ" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The input value for the direction the center of elements should have">Center
          Line Z</label>
        <input type="number" step="any" class="form-control" name="centerLineZ" placeholder="1.0"
          data-3d-input-only="true" />
      </div>
    </div>
    <div class="mb-3">
      <label for="numPoints" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The number of point sources to use when simulating the transducer">Points</label>
      <input type="number" class="form-control" name="numPoints" placeholder="20000" required />
    </div>
    <div class="mb-3">
      <label for="numElements" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The number of elements of the phased array">Elements</label>
      <input type="number" class="form-control" name="numElements" placeholder="16" required />
    </div>
    <div class="mb-3">
      <label for="pitch" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The distance (in meters) between the centers of neighboring elements in the phased array">Pitch</label>
      <input type="number" step="any" class="form-control" name="pitch" placeholder="0.0015" required />
    </div>
    <div class="mb-3">
      <label for="elementWidth" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The width (in meters) of each individual element of the array">Element
        Width</label>
      <input type="number" step="any" class="form-control" name="elementWidth" placeholder="0.0012" required />
    </div>
    <div class="mb-3">
      <label for="tiltAngle" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The desired tilt angle (in degrees) of the wavefront">Tilt Angle</label>
      <input type="number" step="any" class="form-control" name="tiltAngle" placeholder="30" required />
    </div>
    <div class="mb-3">
      <label for="focalLengthCheck" data-bs-toggle="tooltip" data-bs-placement="right">Focused</label>
      <div class="form-check form-switch">
        <input class="form-check-input" type="checkbox" id="focalLengthCheck" name="focalLengthCheck"
          onchange="focalLengthCheckClick(this)" />
      </div>
      <div class="mb-3 ms-3" data-focal-length="true" hidden>
        <label for="focalLength" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The focal length (in meters) of the transducer">Focal length</label>
        <input type="number" step="any" class="form-control" name="focalLength" id="focalLength" placeholder="0" />
      </div>
    </div>
    <div class="mb-3" data-3d-only="true" hidden>
      <label for="height" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The height (in meters) of the elements of the array">Height</label>
      <input type="number" step="any" class="form-control" name="height" placeholder="0.005" data-3d-input-only="true" />
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
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>

<style scoped></style>
