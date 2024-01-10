<template>
  <form @submit.prevent>
    <fieldset>
      <div class="mb-3">
        <label for="sliceAxis" class="form-label" data-bs-toggle="tooltip" data-bs-placement="right"
          title="Axis along which to slice the 3D field to be recorded">Axis</label>
        <select class="form-select" aria-label="Axis" id="sliceAxis" v-model="sliceAxis" v-if="is2d">
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
        <input type="number" step="any" class="form-control" id="slicePosition" v-model="slicePosition" placeholder="0.0"
          v-if="is2d" />
      </div>
    </fieldset>
  </form>
</template>

<script>
export default {
  data() {
    return {
      sliceAxis: '',
      slicePosition: 0.0,
    }
  },
  computed: {
    is2d() {
      return this.$store.getters.is2d
    },
  },
}
</script>

<style scoped></style>
