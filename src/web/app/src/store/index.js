import { createStore } from "vuex";

import { EventBus } from "../event-bus";
import scenarioSettings from "./modules/scenarioSettings";
import simulationSettings from "./modules/simulationSettings";
import displaySettings from "./modules/displaySettings";
import targetSettings from "./modules/targetSettings";
import transducersSettings from "./modules/transducersSettings";
import debounce from "lodash/debounce";

const initialState = () => ({
  hasSimulation: false,
  isRunningSimulation: false,
  isProcessing: false,
  processingMessage: "",

  materials: [],
  materialProperties: [],
  transducerTypes: [],
});

const store = createStore({
  modules: {
    scenarioSettings,
    simulationSettings,
    displaySettings,
    targetSettings,
    transducersSettings,
  },
  state() {
    return {
      is2d: true,
      validity: {
        scenario: false,
        display: false,
        target: false,
        simulation: false,
      },
      ...initialState(),
    };
  },
  mutations: {
    setComponentValidity(state, { component, isValid }) {
      state.validity = {
        ...state.validity,
        [component]: isValid,
      };
    },
    reset(state) {
      Object.assign(state, initialState());
    },
    set2d(state, payload) {
      state.is2d = payload;
    },
    setInitialValues(state, payload) {
      state.hasSimulation = payload.has_simulation;
      state.isRunningSimulation = payload.is_running_simulation;
      state.materials = payload.materials;
      state.materialProperties = payload.material_properties;
      state.transducerTypes = payload.transducer_types;

      this.dispatch("scenarioSettings/setAvailableCTs", payload.available_cts, {
        root: true,
      });
      this.dispatch(
        "scenarioSettings/setBuiltInScenarios",
        payload.built_in_scenarios,
        { root: true }
      );
      if (payload.configuration !== null) {
        state.is2d = payload.configuration.is2d;
        this.dispatch(
          "scenarioSettings/setScenario",
          payload.configuration.scenarioSettings,
          { root: true }
        );
        this.dispatch(
          "displaySettings/setDisplaySettings",
          payload.configuration.displaySettings,
          {
            root: true,
          }
        );
        this.dispatch(
          "transducersSettings/setTransducers",
          payload.configuration.transducers,
          {
            root: true,
          }
        );
        this.dispatch(
          "simulationSettings/setSimulationSettings",
          payload.configuration.simulationSettings,
          {
            root: true,
          }
        );
        this.dispatch(
          "targetSettings/setTarget",
          payload.configuration.target,
          {
            root: true,
          }
        );
      }
      if (state.hasSimulation || state.isRunningSimulation) {
        this.dispatch("getSimulation");
      }
    },
    setHasSimulation(state, payload) {
      state.hasSimulation = payload;
    },
    setRenderedOutput(state, payload) {
      state.renderedOutput = payload;
    },
    setIsRunningSimulation(state, payload) {
      state.isRunningSimulation = payload;
    },
    setProcessingMessage(state, payload) {
      state.processingMessage = payload;
    },
    setIsProcessing(state, payload) {
      state.isProcessing = payload;
    },
  },
  actions: {
    updateValidityState({ commit }, { component, isValid }) {
      commit("setComponentValidity", { component, isValid });
    },
    setIsProcessing({ commit }, payload) {
      commit("setIsProcessing", payload);
    },
    setProcessingMessage({ commit }, payload) {
      commit("setProcessingMessage", payload);
    },
    reset({ commit }) {
      // Reset root state
      commit("setRenderedOutput", null);

      commit("reset");

      // Reset each submodule
      commit("displaySettings/reset", null, { root: true });
      commit("scenarioSettings/reset", null, { root: true });
      commit("simulationSettings/reset", null, { root: true });
      commit("targetSettings/reset", null, { root: true });
      commit("transducersSettings/reset", null, { root: true });
    },
    set2d({ commit, dispatch }, payload) {
      commit("set2d", payload);
      dispatch("reset");
    },
    async getInitialData({ commit }) {
      await fetch(`http://${process.env.VUE_APP_BACKEND_URL}/info`)
        .then((response) => {
          if (response.ok) {
            return response.json();
          }
          throw new Error("Network response was not ok.");
        })
        .then((data) => {
          commit("setInitialValues", data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    async runSimulation({ commit, dispatch }) {
      commit("setRenderedOutput", null);
      commit("setIsRunningSimulation", true);
      commit("setIsProcessing", true);
      commit("setProcessingMessage", "Running simulation...");
      const payload = await dispatch("getPayload");

      await fetch(`http://${process.env.VUE_APP_BACKEND_URL}/simulation`, {
        method: "POST",
        headers: new Headers({
          "Content-Type": "application/json",
        }),
        body: JSON.stringify(payload),
      })
        .then((response) => {
          if (!response.ok) {
            return Promise.reject(response);
          }
          return response.json();
        })
        .then((data) => {
          if (data) {
            commit("setHasSimulation", true);
            commit("setRenderedOutput", data.data);
            commit("setIsRunningSimulation", false);
            commit("setIsProcessing", false);
          }
        })
        .catch((response) => {
          commit("setIsProcessing", false);
          response.json().then((error) => {
            console.error(
              "There was a problem with the fetch operation:",
              error
            );
          });
        });
    },
    async cleanSimulation({ commit }) {
      // call the backend to cancel the simulation
      try {
        const response = await fetch(
          `http://${process.env.VUE_APP_BACKEND_URL}/simulation`,
          {
            method: "DELETE",
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      } catch (error) {
        console.error("There was a problem with the fetch operation:", error);
      }
      commit("setRenderedOutput", null);
      commit("setHasSimulation", false);
      commit("setIsRunningSimulation", false);
      EventBus.emit("reset-parameters-panel");
    },
    getPayload({ state, rootGetters }) {
      const body = {
        is2d: state.is2d,
        scenarioSettings:
          rootGetters["scenarioSettings/scenarioSettingsPayload"],
        transducers: rootGetters["transducersSettings/transducers"],
        target: rootGetters["targetSettings/targetPayload"],
        simulationSettings:
          rootGetters["simulationSettings/simulationSettings"],
        displaySettings: rootGetters["displaySettings/displaySettings"],
      };
      return body;
    },
    async renderLayout({ commit, dispatch, getters }) {
      // Check if we have enough data to render the layout
      if (
        !getters.canRenderLayout ||
        getters.hasSimulation ||
        getters.isRunningSimulation
      )
        return;
      EventBus.emit("validate");
      if (!getters.areAllComponentsValid) {
        return;
      }
      commit("setIsProcessing", true);
      commit("setRenderedOutput", null);
      commit("setProcessingMessage", "Rendering layout...");
      const payload = await dispatch("getPayload");
      try {
        const response = await fetch(
          `http://${process.env.VUE_APP_BACKEND_URL}/render_layout`,
          {
            method: "POST",
            headers: new Headers({
              "Content-Type": "application/json",
            }),
            body: JSON.stringify(payload),
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        commit("setRenderedOutput", data.data);
        commit("setIsProcessing", false);
      } catch (error) {
        commit("setIsProcessing", false);
        console.error("There was a problem with the fetch operation:", error);
      }
    },
    async getSimulation({ commit }) {
      commit("setRenderedOutput", null);
      commit("setIsRunningSimulation", true);
      commit("setIsProcessing", true);
      commit("setProcessingMessage", "Running simulation...");
      try {
        const response = await fetch(
          `http://${process.env.VUE_APP_BACKEND_URL}/simulation`,
          {
            method: "GET",
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        commit("setRenderedOutput", data.data);
        commit("setHasSimulation", true);
        commit("setIsRunningSimulation", false);
        commit("setIsProcessing", false);
      } catch (error) {
        console.error("There was a problem with the fetch operation:", error);
        commit("setIsProcessing", false);
      }
    },
  },
  getters: {
    areAllComponentsValid(state) {
      for (const key in state.validity) {
        if (!state.validity[key]) {
          if (key === "display" && state.is2d) {
            continue;
          }
          EventBus.emit("open-panel", key);
          return false;
        }
      }
      return true;
    },
    canRenderLayout(state, getters) {
      return getters["scenarioSettings/isScenarioValid"];
    },
    canRunSimulation(state, getters, _rootState, rootGetters) {
      const isScenarioValid = getters["scenarioSettings/isScenarioValid"];
      const transducers =
        rootGetters["transducersSettings/transducers"].length > 0;
      return isScenarioValid && transducers && !state.isRunningSimulation;
    },
    renderedOutput(state) {
      return state.renderedOutput;
    },
    isRunningSimulation(state) {
      return state.isRunningSimulation;
    },
    isProcessing(state) {
      return state.isProcessing;
    },
    processingMessage(state) {
      return state.processingMessage;
    },
    hasSimulation(state) {
      return state.hasSimulation;
    },
    is2d(state) {
      return state.is2d;
    },
  },
});

const debouncedRenderLayout = debounce(() => {
  store.dispatch("renderLayout");
}, 500);

store.subscribe((mutation) => {
  const namespaces = [
    "displaySettings/",
    "targetSettings/",
    "transducersSettings/",
    "simulationSettings/",
    "scenarioSettings/",
  ];

  const isNamespaceMutation = namespaces.some((namespace) =>
    mutation.type.startsWith(namespace)
  );

  if (isNamespaceMutation) {
    debouncedRenderLayout();
  }
});

export default store;
