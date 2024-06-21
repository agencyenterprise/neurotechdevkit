<template>
  <div class="mb-3">
    <label class="form-label" title="Axis along which to slice the 3D field to be recorded">Axis</label>
    <select :disabled="hasSimulation" class="form-select" aria-label="Axis" v-model="sliceAxis">
      <option disabled value="">Select an axis</option>
      <option value="x">X</option>
      <option value="y">Y</option>
      <option value="z">Z</option>
    </select>
  </div>
  <div class="mb-3">
    <label class="form-label"
      title="The position (in meters) along the slice axis at which the slice of the 3D field should be made">Distance
      from Origin (m)</label>
    <input :disabled="hasSimulation" type="number" step="any" class="form-control" v-model.number="slicePosition"
      placeholder="0.0" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  computed: {
    ...mapGetters(['is2d', 'hasSimulation']),
    sliceAxis: {
      get() {
        return this.$store.getters['displaySettings/sliceAxis'];
      },
      set(value) {
        this.$store.dispatch('displaySettings/setSliceAxis', value);
      }
    },
    slicePosition: {
      get() {
        return this.$store.getters['displaySettings/slicePosition'];
      },
      set(value) {
        this.$store.dispatch('displaySettings/setSlicePosition', value);
      }
    },
  },
};
</script>

<style scoped></style>