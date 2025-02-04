<template>
  <form ref="form" @input="onInputChanged">
    <div class="mb-3">
      <label class="form-label" v-tooltip="'Axis along which to slice the 3D field to be recorded'">Axis</label>
      <select :disabled="hasSimulation || isRunningSimulation" class="form-select" aria-label="Axis" v-model="sliceAxis"
        required>
        <option disabled value="">Select an axis</option>
        <option value="x">X</option>
        <option value="y">Y</option>
        <option value="z">Z</option>
      </select>
    </div>
    <div class="mb-3">
      <label class="form-label"
        v-tooltip="'The position (in meters) along the slice axis at which the slice of the 3D field should be made'">Distance
        from origin (m)</label>
      <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control"
        v-model.number="slicePosition" placeholder="0.0" required />
    </div>
  </form>
</template>

<script>
import { EventBus } from '../event-bus';
import { mapGetters, mapActions } from 'vuex';

export default {
  beforeUnmount() {
    EventBus.off('validate', this.validate);
  },
  mounted() {
    EventBus.on('validate', this.validate);
  },
  methods: {
    ...mapActions(['updateValidityState']),
    validate() {
      const isValid = this.$refs.form.checkValidity();
      this.updateValidityState({ component: 'display', isValid });
      return isValid;
    },
    onInputChanged() {
      if (!this.validate()) {
        this.$refs.form.reportValidity();
      }
    }
  },
  computed: {
    ...mapGetters(['is2d', 'hasSimulation', 'isRunningSimulation']),
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