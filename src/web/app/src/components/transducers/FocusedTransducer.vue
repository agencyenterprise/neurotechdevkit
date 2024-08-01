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
          v-model.number="positionX" required />
      </div>
      <div class="col mb-3">
        <label v-tooltip="'The coordinate (in meters) of the point at the center of the transducer'">Position
          Y</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0"
          v-model.number="positionY" required />
      </div>
      <div class="col mb-3" v-if="!is2d">
        <label v-tooltip="'The coordinate (in meters) of the point at the center of the transducer'">Position
          Z</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d"
          v-model.number="positionZ" required />
      </div>
    </div>
    <div class="row">
      <div class="col mb-3">
        <label v-tooltip="'Indicate the direction the source is pointing'">Direction X</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0"
          v-model.number="directionX" required />
      </div>
      <div class="col mb-3">
        <label v-tooltip="'Indicate the direction the source is pointing'">Direction Y</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="1.0"
          v-model.number="directionY" required />
      </div>
      <div class="col mb-3" v-if="!is2d">
        <label v-tooltip="'Indicate the direction the source is pointing'">Direction Z</label>
        <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d"
          v-model.number="directionZ" required />
      </div>
    </div>
    <div class="mb-3">
      <label v-tooltip="'The aperture (in meters) of the transducer'">Aperture</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.064"
        v-model.number="aperture" required />
    </div>
    <div class="mb-3">
      <label v-tooltip="'The focal length (in meters) of the transducer'">Focal length</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.064"
        v-model.number="focalLength" required />
    </div>
    <div class="mb-3">
      <label v-tooltip="'The number of point sources to use when simulating the transducer'">Points</label>
      <input :disabled="readOnly" type="number" class="form-control" placeholder="20000" v-model.number="numPoints"
        required />
    </div>
    <div class="mb-3">
      <label v-tooltip="'The delay (in seconds) that the source will wait before emitting'">Delay</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" placeholder="0.0" v-model.number="delay"
        required />
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
        directionX: 0.0,
        directionY: 1.0,
        directionZ: 0.0,
        aperture: 0.064,
        focalLength: 0.064,
        numPoints: 20000,
        delay: 0.0,
      },
      positionX: null,
      positionY: null,
      positionZ: null,
      directionX: null,
      directionY: null,
      directionZ: null,
      aperture: null,
      focalLength: null,
      numPoints: null,
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
      return {
        position,
        direction,
        aperture: this.aperture,
        focalLength: this.focalLength,
        numPoints: this.numPoints,
        delay: this.delay,
        transducerType: 'focusedSource',
      };
    },
    setTransducerSettings(settings) {
      this.positionX = settings.position[0];
      this.positionY = settings.position[1];
      this.positionZ = settings.position[2] || 0.0;
      this.directionX = settings.direction[0];
      this.directionY = settings.direction[1];
      this.directionZ = settings.direction[2] || 0.0;
      this.aperture = settings.aperture;
      this.focalLength = settings.focalLength;
      this.numPoints = settings.numPoints;
      this.delay = settings.delay;
    },
    fillDefaultValues() {
      // Automatically set each field to its default value
      for (const key in this.defaultValues) {
        this[key] = this.defaultValues[key];
      }
    }
  },
}
</script>
<style scoped></style>
