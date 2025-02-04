<template>
  <div class="row">
    <div class="col text-end">
      <button :disabled="readOnly" type="button" class="btn btn-link" @click="fillDefaultValues">
        Fill default values
      </button>
    </div>
  </div>
  <form ref="form" @input="validateForm">
    <div class="row">
      <div class="col mb-3">
        <label v-tooltip="'The coordinate (in meters) of the point at the center of the transducer'">Position
          X</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0"
          v-model.number="positionX" required/>
      </div>
      <div class="col mb-3">
        <label v-tooltip="'The coordinate (in meters) of the point at the center of the transducer'">Position
          Y</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0"
          v-model.number="positionY" required/>
      </div>
      <div class="col mb-3" v-if="!is2d">
        <label v-tooltip="'The coordinate (in meters) of the point at the center of the transducer'">Position
          Z</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d"
          v-model.number="positionZ" required/>
      </div>
    </div>
    <div class="row">
      <div class="col mb-3">
        <label v-tooltip="'Indicate the direction the source is pointing'">Direction X</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="1.0"
          v-model.number="directionX" required/>
      </div>
      <div class="col mb-3">
        <label v-tooltip="'Indicate the direction the source is pointing'">Direction Y</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0"
          v-model.number="directionY" required/>
      </div>
      <div class="col mb-3" v-if="!is2d">
        <label v-tooltip="'Indicate the direction the source is pointing'">Direction Z</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d"
          v-model.number="directionZ" required/>
      </div>
    </div>
    <div class="row" v-if="!is2d">
      <div class="col mb-3">
        <label v-tooltip="'The input value for the direction the center of elements should have'">Center
          Line X</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d"
          v-model.number="centerX" required/>
      </div>
      <div class="col mb-3">
        <label v-tooltip="'The input value for the direction the center of elements should have'">Center
          Line Y</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d"
          v-model.number="centerY" required/>
      </div>
      <div class="col mb-3">
        <label v-tooltip="'The input value for the direction the center of elements should have'">Center
          Line Z</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="1.0" v-if="!is2d"
          v-model.number="centerZ" required/>
      </div>
    </div>
    <div class="mb-3">
      <label v-tooltip="'The number of point sources to use when simulating the transducer'">Points</label>
      <input :disabled="readOnly" type="number" class="form-control" placeholder="20000" v-model.number="numPoints" required/>
    </div>
    <div class="mb-3">
      <label v-tooltip="'The number of elements of the phased array'">Elements</label>
      <input :disabled="readOnly" type="number" class="form-control" placeholder="16" v-model.number="numElements" required/>
    </div>
    <div class="mb-3">
      <label
        v-tooltip="'The distance (in meters) between the centers of neighboring elements in the phased array'">Pitch</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0015"
        v-model.number="pitch" required/>
    </div>
    <div class="mb-3">
      <label v-tooltip="'The width (in meters) of each individual element of the array'">Element width</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0012"
        v-model.number="elementWidth" required/>
    </div>
    <div class="mb-3">
      <label v-tooltip="'The desired tilt angle (in degrees) of the wavefront'">Tilt angle</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="30"
        v-model.number="tiltAngle" required/>
    </div>
    <div class="mb-3">
      <label>Focused</label>
      <div class="form-check form-switch">
        <input :disabled="readOnly" class="form-check-input" type="checkbox" id="focalLengthCheck"
          v-model="focalLengthEnabled" />
      </div>
      <div class="mb-3 ms-3" v-if="focalLengthEnabled">
        <label v-tooltip="'The focal length (in meters) of the transducer'">Focal length</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" id="focalLength" placeholder="0"
          v-model.number="focalLength" required/>
      </div>
    </div>
    <div class="mb-3" v-if="!is2d">
      <label v-tooltip="'The height (in meters) of the elements of the array'">Height</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.005" v-if="!is2d"
        v-model.number="height" required/>
    </div>
    <div class="mb-3">
      <label v-tooltip="'The delay (in seconds) that the source will wait before emitting'">Delay</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0"
        v-model.number="delay" required/>
    </div>
  </form>
</template>

<script>
export default {
  props: {
    readOnly: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      defaultValues: {
        positionX: 0.0,
        positionY: 0.0,
        positionZ: 0.0,
        directionX: 1.0,
        directionY: 0.0,
        directionZ: 0.0,
        centerX: 0.0,
        centerY: 0.0,
        centerZ: 1.0,
        numPoints: 20000,
        numElements: 16,
        pitch: 0.0015,
        elementWidth: 0.0012,
        tiltAngle: 30,
        focalLength: 0,
        height: 0.005,
        delay: 0.0,
      },
      focalLengthEnabled: false,
      positionX: null,
      positionY: null,
      positionZ: null,
      directionX: null,
      directionY: null,
      directionZ: null,
      centerX: null,
      centerY: null,
      centerZ: null,
      numPoints: null,
      numElements: null,
      pitch: null,
      elementWidth: null,
      tiltAngle: null,
      focalLength: null,
      height: null,
      delay: null,
    };
  },
  computed: {
    is2d() {
      return this.$store.getters.is2d
    },
  },
  methods: {
    validateForm() {
      if (!this.$refs.form.checkValidity()) {
        this.$refs.form.reportValidity();
        return false;
      }
      return true;
    },
    getTransducerSettings() {
      let position = [this.positionX, this.positionY]
      if (!this.is2d) {
        position.push(this.positionZ)
      }
      let direction = [this.directionX, this.directionY]
      if (!this.is2d) {
        direction.push(this.directionZ)
      }
      const payload = {
        position,
        direction,
        numPoints: this.numPoints,
        numElements: this.numElements,
        pitch: this.pitch,
        elementWidth: this.elementWidth,
        tiltAngle: this.tiltAngle,
        focalLength: this.focalLength,
        height: this.height,
        delay: this.delay,
        transducerType: 'phasedArraySource',
      }
      if (!this.is2d) {
        let center = [this.centerX, this.centerY, this.centerZ]
        payload.center = center
      }
      return payload;
    },
    setTransducerSettings(settings) {
      this.positionX = settings.position[0];
      this.positionY = settings.position[1];
      this.positionZ = settings.position[2] || 0.0;
      this.directionX = settings.direction[0];
      this.directionY = settings.direction[1];
      this.directionZ = settings.direction[2] || 0.0;
      if (settings.center) {
        this.centerX = settings.center[0];
        this.centerY = settings.center[1];
        this.centerZ = settings.center[2] || 0.0;
      }
      this.numPoints = settings.numPoints;
      this.numElements = settings.numElements;
      this.pitch = settings.pitch;
      this.elementWidth = settings.elementWidth;
      this.tiltAngle = settings.tiltAngle;
      this.focalLengthEnabled = settings.focalLength > 0;
      this.focalLength = settings.focalLength;
      this.height = settings.height;
      this.delay = settings.delay;
    },
    fillDefaultValues() {
      // Automatically set each field to its default value
      for (const key in this.defaultValues) {
        this[key] = this.defaultValues[key];
      }
    },
  },
}
</script>

<style scoped></style>
