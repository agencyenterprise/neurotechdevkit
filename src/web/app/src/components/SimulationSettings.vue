<template>
  <div class="mb-3">
    <label title="The number of points per wavelength">Simulation precision</label>
    <input :disabled="hasSimulation" type="number" v-model.number="simulationPrecision" class="form-control" />
  </div>
  <div class="mb-3">
    <label title="The center frequency of the transducers (in Hz)">Center frequency</label>
    <input :disabled="hasSimulation" type="number" v-model.number="centerFrequency" class="form-control" />
  </div>

  <div class="mb-3">
    <label>Material properties</label>
    <MaterialsSettings />
  </div>
  <div class="mb-3">
    <label>Simulation type</label>
    <div class="form-control">
      <div class="form-check">
        <input :disabled="hasSimulation" class="form-check-input" type="radio" id="pulsed" value="false"
          v-model="isSteadySimulation" />
        <label class="form-check-label" for="pulsed"
          title="The transducers will emit a pulse containing a few cycles of oscillation and then let the pulse propagate out to all edges of the scenario">
          Pulsed
        </label>
      </div>
      <div class="form-check">
        <input :disabled="hasSimulation" class="form-check-input" type="radio" id="steadyState" value="true"
          v-model="isSteadySimulation" />
        <label class="form-check-label" for="steadyState"
          title="The transducers will emit pressure waves with a continuous waveform until steady-state has been reached">
          Steady state
        </label>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import MaterialsSettings from './MaterialsSettings.vue';

export default {
  components: {
    MaterialsSettings
  },
  computed: {
    ...mapGetters(['hasSimulation']),
    centerFrequency: {
      get() {
        return this.$store.getters['simulationSettings/centerFrequency'];
      },
      set(value) {
        this.$store.dispatch('simulationSettings/setCenterFrequency', value);
      }
    },
    simulationPrecision: {
      get() {
        return this.$store.getters['simulationSettings/simulationPrecision'];
      },
      set(value) {
        this.$store.dispatch('simulationSettings/setSimulationPrecision', value);
      }
    },
    isSteadySimulation: {
      get() {
        return this.$store.state.simulationSettings.isSteadySimulation;
      },
      set(value) {
        this.$store.commit('simulationSettings/setIsSteadySimulation', value === 'true');
      }
    }
  }
}
</script>

<style scoped></style>