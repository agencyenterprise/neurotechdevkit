<template>
  <div class="row">
    <div class="col mb-3">
      <label title="The coordinate (in meters) of the point at the center of the transducer">Position
        X</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-model.number="positionX" />
    </div>
    <div class="col mb-3">
      <label title="The coordinate (in meters) of the point at the center of the transducer">Position
        Y</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-model.number="positionY" />
    </div>
    <div class="col mb-3" v-if="!is2d">
      <label title="The coordinate (in meters) of the point at the center of the transducer">Position
        Z</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d" v-model.number="positionZ" />
    </div>
  </div>
  <div class="row">
    <div class="col mb-3">
      <label title="Indicate the direction the source is pointing">Direction X</label>
      <input type="number" step="any" class="form-control" placeholder="1.0" v-model.number="directionX" />
    </div>
    <div class="col mb-3">
      <label title="Indicate the direction the source is pointing">Direction Y</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-model.number="directionY" />
    </div>
    <div class="col mb-3" v-if="!is2d">
      <label title="Indicate the direction the source is pointing">Direction Z</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d" v-model.number="directionZ" />
    </div>
  </div>
  <div class="row" v-if="!is2d">
    <div class="col mb-3">
      <label title="The input value for the direction the center of elements should have">Center
        Line X</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d" v-model.number="centerX" />
    </div>
    <div class="col mb-3">
      <label title="The input value for the direction the center of elements should have">Center
        Line Y</label>
      <input type="number" step="any" class="form-control" placeholder="0.0" v-if="!is2d" v-model.number="centerY" />
    </div>
    <div class="col mb-3">
      <label title="The input value for the direction the center of elements should have">Center
        Line Z</label>
      <input type="number" step="any" class="form-control" placeholder="1.0" v-if="!is2d" v-model.number="centerZ" />
    </div>
  </div>
  <div class="mb-3">
    <label title="The number of point sources to use when simulating the transducer">Points</label>
    <input type="number" class="form-control" placeholder="20000" v-model.number="numPoints" />
  </div>
  <div class="mb-3">
    <label title="The number of elements of the phased array">Elements</label>
    <input type="number" class="form-control" placeholder="16" v-model.number="elements" />
  </div>
  <div class="mb-3">
    <label
      title="The distance (in meters) between the centers of neighboring elements in the phased array">Pitch</label>
    <input type="number" step="any" class="form-control" placeholder="0.0015" v-model.number="pitch" />
  </div>
  <div class="mb-3">
    <label title="The width (in meters) of each individual element of the array">Element width</label>
    <input type="number" step="any" class="form-control" placeholder="0.0012" v-model.number="elementWidth" />
  </div>
  <div class="mb-3">
    <label title="The desired tilt angle (in degrees) of the wavefront">Tilt angle</label>
    <input type="number" step="any" class="form-control" placeholder="30" v-model.number="tiltAngle" />
  </div>
  <div class="mb-3">
    <label>Focused</label>
    <div class="form-check form-switch">
      <input class="form-check-input" type="checkbox" id="focalLengthCheck" />
    </div>
    <div class="mb-3 ms-3" data-focal-length="true">
      <label title="The focal length (in meters) of the transducer">Focal length</label>
      <input type="number" step="any" class="form-control" id="focalLength" placeholder="0"
        v-model.number="focalLength" />
    </div>
  </div>
  <div class="mb-3" v-if="!is2d">
    <label title="The height (in meters) of the elements of the array">Height</label>
    <input type="number" step="any" class="form-control" placeholder="0.005" v-if="!is2d" v-model.number="height" />
  </div>
  <div class="mb-3">
    <label title="The delay (in seconds) that the source will wait before emitting">Delay</label>
    <input type="number" step="any" class="form-control" placeholder="0.0" v-model.number="delay" />
  </div>
</template>

<script>
export default {
  data() {
    return {
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
      elements: 16,
      pitch: 0.0015,
      elementWidth: 0.0012,
      tiltAngle: 30,
      focalLength: 0,
      height: 0.005,
      delay: 0.0,
    };
  },
  computed: {
    is2d() {
      return this.$store.getters.is2d
    },
  },
  methods: {
    getTransducerSettings() {
      let position = [this.positionY, this.positionX]
      if (!this.is2d) {
        position.push(this.positionZ)
      }
      let direction = [this.directionY, this.directionX]
      if (!this.is2d) {
        direction.push(this.directionZ)
      }
      let center = [this.centerY, this.centerX]
      if (!this.is2d) {
        center.push(this.centerZ)
      }
      return {
        position,
        direction,
        center,
        numPoints: this.numPoints,
        elements: this.elements,
        pitch: this.pitch,
        elementWidth: this.elementWidth,
        tiltAngle: this.tiltAngle,
        focalLength: this.focalLength,
        height: this.height,
        delay: this.delay,
        transducerType: 'phasedArraySource',
      };
    },

    setTransducerSettings(settings) {
      this.positionX = settings.position[1];
      this.positionY = settings.position[0];
      this.positionZ = settings.position[2] || 0.0;
      this.directionX = settings.direction[1];
      this.directionY = settings.direction[0];
      this.directionZ = settings.direction[2] || 0.0;
      this.centerX = settings.center[1];
      this.centerY = settings.center[0];
      this.centerZ = settings.center[2] || 0.0;
      this.numPoints = settings.numPoints;
      this.elements = settings.elements;
      this.pitch = settings.pitch;
      this.elementWidth = settings.elementWidth;
      this.tiltAngle = settings.tiltAngle;
      this.focalLength = settings.focalLength;
      this.height = settings.height;
      this.delay = settings.delay;
    }
  },
}
</script>

<style scoped></style>
