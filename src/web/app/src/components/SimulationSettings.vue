<template>
  <form id="simulationSettingsForm">
    <fieldset>
      <div class="mb-3">
        <label for="simulationPrecision" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The number of points per wavelength">Simulation Precision</label>
        <input type="number" class="form-control" name="simulationPrecision" id="simulationPrecision" required disabled />
      </div>
      <div class="mb-3">
        <label for="centerFrequency" data-bs-toggle="tooltip" data-bs-placement="right"
          title="The center frequency of the transducers (in Hz)">Center Frequency</label>
        <input type="number" class="form-control" name="centerFrequency" id="centerFrequency" onchange="valueChanged()"
          required />
      </div>
    </fieldset>
  </form>
  <div class="mb-3">
    <label for="materialsAccordion">Material properties</label>
    <MaterialsSettings />
  </div>
  <div class="mb-3">
    <label for="simulationType">Simulation Type</label>
    <fieldset>
      <div id="simulationType" class="form-control">
        <div class="form-check">
          <input class="form-check-input" type="radio" name="flexRadioDefault" id="simulationTypePulsed" />
          <label class="form-check-label" for="simulationTypePulsed" data-bs-toggle="tooltip" data-bs-placement="right"
            title="The transducers will emit a pulse containing a few cycles of oscillation and then let the pulse propagate out to all edges of the scenario">
            Pulsed
          </label>
        </div>
        <div class="form-check">
          <input class="form-check-input" type="radio" name="flexRadioDefault" id="simulationTypeSteady" checked />
          <label class="form-check-label" for="simulationTypeSteady" data-bs-toggle="tooltip" data-bs-placement="right"
            title="The transducers will emit pressure waves with a continuous waveform until steady-state has been reached">
            Steady State
          </label>
        </div>
      </div>
    </fieldset>
  </div>
</template>

<script>
import MaterialsSettings from './MaterialsSettings.vue'
export default {
  components: {
    MaterialsSettings
  },
}
</script>

<style scoped></style>
