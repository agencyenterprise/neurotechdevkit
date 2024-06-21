<template>
  <div class="mb-3 row">
    <div class="col">
      <label for="centerX" title="The location of the center of the target (in meters)">Center X</label>
      <input :disabled="hasSimulation" type="number" step="any" class="form-control" id="centerX" v-model.number="centerX" />
    </div>
    <div class="col">
      <label for="centerY" title="The location of the center of the target (in meters)">Center Y</label>
      <input :disabled="hasSimulation" type="number" step="any" class="form-control" id="centerY" v-model.number="centerY" />
    </div>
    <div class="col" v-if="!is2d">
      <label for="centerZ" title="The location of the center of the target (in meters)">Center Z</label>
      <input :disabled="hasSimulation" type="number" step="any" class="form-control" id="centerZ"
        v-model.number="centerZ" />
    </div>
  </div>
  <div class="mb-3">
    <label for="radius" title="The radius of the target (in meters)">Radius</label>
    <input :disabled="hasSimulation" type="number" step="any" class="form-control" id="radius" v-model.number="radius" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  computed: {
    ...mapGetters(['is2d', 'hasSimulation']),
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
    },
  },

};
</script>

<style scoped></style>