<template>
  <form ref="form" @input="onInputChanged">
    <div class="mb-3 row">
      <div class="col">
        <label for="centerX" v-tooltip="'The location of the center of the target (in meters)'">Center X</label>
        <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control"
          placeholder="0.0" id="centerX" v-model.number="centerX" required>
      </div>
      <div class="col">
        <label for="centerY" v-tooltip="'The location of the center of the target (in meters)'">Center Y</label>
        <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control"
          placeholder="0.0" id="centerY" v-model.number="centerY" required />
      </div>
      <div class="col" v-if="!is2d">
        <label for="centerZ" v-tooltip="'The location of the center of the target (in meters)'">Center Z</label>
        <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control"
          placeholder="0.0" id="centerZ" v-model.number="centerZ" required />
      </div>
    </div>
    <div class="mb-3">
      <label for="radius" v-tooltip="'The radius of the target (in meters)'">Radius</label>
      <input :disabled="hasSimulation || isRunningSimulation" type="number" step="any" class="form-control" id="radius"
        placeholder="0.01" v-model.number="radius" required />
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
      this.updateValidityState({ component: 'target', isValid });
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
    centerX: {
      get() {
        return this.$store.getters['targetSettings/centerX'];
      },
      set(value) {
        this.$store.dispatch('targetSettings/updateTargetProperty', { key: 'centerX', value });
      }
    },
    centerY: {
      get() {
        return this.$store.getters['targetSettings/centerY'];
      },
      set(value) {
        this.$store.dispatch('targetSettings/updateTargetProperty', { key: 'centerY', value });
      }
    },
    centerZ: {
      get() {
        return this.$store.getters['targetSettings/centerZ'];
      },
      set(value) {
        this.$store.dispatch('targetSettings/updateTargetProperty', { key: 'centerZ', value });
      }
    },
    radius: {
      get() {
        return this.$store.getters['targetSettings/radius'];
      },
      set(value) {
        this.$store.dispatch('targetSettings/updateTargetProperty', { key: 'radius', value });
      }
    }
  },
};
</script>

<style scoped></style>