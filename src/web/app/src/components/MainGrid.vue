<template>
  <div class="image-container">
    <img v-if="renderedImage" :src="renderedImage" alt="Rendered Layout" />

    <!-- Run Simulation Button -->
    <button v-show="showRunButton" @click="runSimulation" :disabled="!canRunSimulation" class="simulation-button">
      Run Simulation
    </button>

    <!-- Cancel Simulation Button -->
    <button v-show="showCancelButton" @click="cancelSimulation" class="simulation-button">
      Cancel Simulation
    </button>

    <!-- Clean Simulation Button -->
    <button v-show="showCleanButton" @click="cleanSimulation" class="simulation-button">
      Clean Simulation
    </button>

    <div v-if="isRunningSimulation" class="simulation-message">
      Running simulation...
    </div>
  </div>
</template>

<script>
import { mapGetters, mapActions } from 'vuex';

export default {
  name: 'MainGrid',
  computed: {
    ...mapGetters(['renderedOutput', 'hasSimulation', 'isRunningSimulation', 'canRunSimulation']),

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
}

.simulation-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 10;
  color: white;
  font-size: 1.5em;
  background-color: rgba(0, 0, 0, 0.7);
  padding: 10px;
  border-radius: 5px;
}
</style>