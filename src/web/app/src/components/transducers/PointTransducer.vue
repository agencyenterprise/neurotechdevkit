<template>
  <div class="row">
    <div class="col mb-3">
      <label for="positionX" title="The coordinate (in meters) of the point at the center of the transducer">Position
        X</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" id="positionX" placeholder="0.0"
        v-model.number="positionX" />
    </div>
    <div class="col mb-3">
      <label for="positionY" title="The coordinate (in meters) of the point at the center of the transducer">Position
        Y</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" id="positionY" placeholder="0.0"
        v-model.number="positionY" />
    </div>
    <div class="col mb-3" v-if="!is2d">
      <label for="positionZ" title="The coordinate (in meters) of the point at the center of the transducer">Position
        Z</label>
      <input :disabled="readOnly" type="number" step="any" class="form-control" id="positionZ" placeholder="0.0"
        v-model.number="positionZ" />
    </div>
  </div>
  <div class="mb-3">
    <label for="delay" title="The delay (in seconds) that the source will wait before emitting">Delay</label>
    <input :disabled="readOnly" type="number" step="any" class="form-control" id="delay" placeholder="0.0"
      v-model.number="delay" />
  </div>
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
      positionX: 0.0,
      positionY: 0.0,
      positionZ: 0.0,
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
      let position = [this.positionY, this.positionX];
      if (!this.is2d) {
        position.push(this.positionZ);
      }
      return {
        position: position,
        delay: this.delay,
        transducerType: 'pointSource',
      };
    },
    setTransducerSettings(settings) {
      this.positionX = settings.position[1];
      this.positionY = settings.position[0];
      this.positionZ = settings.position[2] || 0.0;
      this.delay = settings.delay;
    },

  },
}
</script>


<style scoped></style>