<template>
  <div class="image-container">
    <img v-if="renderedImage" :src="renderedImage" alt="Rendered Layout" />

    <!-- Run Simulation Button -->
    <button v-show="showRunButton" @click="runSimulation" :disabled="!canRunSimulation" class="simulation-button">
      Run simulation
    </button>

    <!-- Cancel Simulation Button -->
    <button v-show="showCancelButton" @click="cancelSimulation" class="simulation-button">
      Cancel simulation
    </button>

    <!-- Clean Simulation Button -->
    <button v-show="showCleanButton" @click="cleanSimulation" class="simulation-button">
      Clean simulation
    </button>
    <div v-if="isProcessing" class="simulation-message">
      <!-- Spinner -->
      <div class="spinner-border" role="status" style="width: 10rem; height: 10rem"></div>
      <!-- Text Message -->
      <span class="simulation-text">{{ processingMessage }}</span>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';

export default {
  name: 'MainGrid',
  computed: {
    ...mapGetters(['renderedOutput', 'hasSimulation', 'isRunningSimulation', 'isProcessing', 'canRunSimulation', 'processingMessage']),

    showRunButton() {
      return !this.isRunningSimulation && !this.hasSimulation;
    },
    showCancelButton() {
      return this.isRunningSimulation;
    },
    showCleanButton() {
      return !this.isRunningSimulation && this.hasSimulation;
    },
    renderedImage() {
      return this.renderedOutput ? `data:image/png;base64,${this.renderedOutput}` : null;
    }
  },
  methods: {
    ...mapActions(['runSimulation', 'cancelSimulation', 'cleanSimulation']),
  }
}
</script>

<style scoped>
.image-container {
  display: flex;
  justify-content: center;
  position: relative;
  height: 100vh;
  width: 100%;
}

.image-container img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.simulation-button {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 10;
  padding: 0.5rem 1rem;
  /* Standard padding for buttons */
  font-size: 1rem;
  /* Standard font size for buttons */
  color: #fff;
  /* Text color for buttons */
  background-color: #007bff;
  /* Primary button color */
  border: none;
  /* Remove default borders */
  border-radius: 0.25rem;
  /* Standard border radius for buttons */
  cursor: pointer;
  /* Indicate that the element is clickable */
  transition: background-color 0.2s, box-shadow 0.2s;
  /* Smooth transitions for interactions */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  /* Subtle shadow for depth */
}

.simulation-button:hover {
  background-color: #0056b3;
  /* Darken button color on hover */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  /* Increase shadow for a "lifted" effect */
}

.simulation-button:disabled {
  background-color: #e0e0e0;
  /* Disabled button color */
  color: #a1a1a1;
  /* Disabled text color */
  cursor: not-allowed;
  box-shadow: none;
  /* No shadow for disabled buttons */
}

.simulation-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 10;
  color: black;
  font-size: 1.5em;
  padding: 10px;
  border-radius: 5px;
  display: flex;
  flex-direction: column;
  /* Stack items vertically */
  align-items: center;
  /* Center spinner and text horizontally */
  justify-content: center;
  /* Center spinner and text vertically */
}

.simulation-text {
  margin-top: 20px;
  /* Add some space between the spinner and the text */
}
</style>