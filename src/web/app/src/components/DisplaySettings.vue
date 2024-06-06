<template>
  <fieldset>
    <div class="mb-3" v-if="is2d">
      <label for="sliceAxis" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
        title="Axis along which to slice the 3D field to be recorded">Axis</label>
      <select class="form-select" aria-label="Axis" id="sliceAxis" v-model="sliceAxis">
        <option value="">Select an axis</option>
        <option value="x">X</option>
        <option value="y">Y</option>
        <option value="z">Z</option>
      </select>
    </div>
    <div class="mb-3" v-if="is2d">
      <label for="slicePosition" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
        title="The position (in meters) along the slice axis at which the slice of the 3D field should be made">Distance
        from Origin (m)</label>
      <input type="number" step="any" class="form-control" id="slicePosition" v-model.number="slicePosition"
        placeholder="0.0" />
    </div>
  </fieldset>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  data() {
    return {
      // If these properties are supposed to be part of Vuex state, you should use computed properties with getters and setters.
      sliceAxis: '',
      slicePosition: 0.0,
    };
  },
  computed: {
    ...mapGetters({
      is2d: 'is2d'
    }),
    // If sliceAxis and slicePosition are part of Vuex, add computed properties with getters and setters here.
  },
  // If you have Vuex actions to update sliceAxis and slicePosition, map them in methods.
  methods: {
    // ...mapActions({
    //   updateSliceAxis: 'updateSliceAxis',
    //   updateSlicePosition: 'updateSlicePosition'
    // }),
  },
  // If you use computed properties with setters, you might not need watchers. If you still need to react to changes, you can add watchers here.
}
</script>

<style scoped>
/* Your styles here */
</style>