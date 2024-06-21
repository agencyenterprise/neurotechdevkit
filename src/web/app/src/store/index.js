import { createStore } from "vuex";

import scenarioSettings from "./modules/scenarioSettings";
import simulationSettings from "./modules/simulationSettings";
import displaySettings from "./modules/displaySettings";
import targetSettings from "./modules/targetSettings";
import transducersSettings from "./modules/transducersSettings";
import debounce from "lodash/debounce";

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
      hasSimulation: false,
      isRunningSimulation: false,

      configuration: {},
      materials: [],
      materialProperties: [],
      transducerTypes: [],
    };
  },
  mutations: {
    set2d(state, payload) {
      state.is2d = payload;
    },
    setInitialValues(state, payload) {
      state.hasSimulation = payload.has_simulation;
      state.isRunningSimulation = payload.is_running_simulation;
      state.configuration = payload.configuration;
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
  },
  actions: {
    set2d(context, payload) {
      context.commit("set2d", payload);
    },
    setInitialValues(context, payload) {
      context.commit("setInitialValues", payload);
    },
    async runSimulation({ commit, dispatch }) {
      commit("setRenderedOutput", null);
      commit("setIsRunningSimulation", true);
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
          }
        })
        .catch((response) => {
          response.json().then((error) => {
            console.error(
              "There was a problem with the fetch operation:",
              error
            );
          });
        });
    },
    async cleanSimulation() {
      await this.dispatch("cancelSimulation");
    },
    async cancelSimulation({ commit }) {
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
    },
    getPayload({ state, rootGetters }) {
      const body = {
        is2d: state.is2d,
        scenarioSettings: {
          isPreBuilt: rootGetters["scenarioSettings/isPreBuilt"],
          scenarioId: rootGetters["scenarioSettings/scenarioId"],
        },
        transducers: rootGetters["transducersSettings/transducers"],
        target: rootGetters["targetSettings/target"],
        simulationSettings:
          rootGetters["simulationSettings/simulationSettings"],
        displaySettings: rootGetters["displaySettings/displaySettings"],
      };
      return body;
    },
    async renderLayout({ commit, dispatch }) {
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
      } catch (error) {
        console.error("There was a problem with the fetch operation:", error);
      }
    },
  },
  getters: {
    canRunSimulation(state, _getters, _rootState, rootGetters) {
      const scenario = rootGetters["scenarioSettings/scenario"];
      return scenario && !state.isRunningSimulation;
    },
    renderedOutput(state) {
      return state.renderedOutput;
    },
    isRunningSimulation(state) {
      return state.isRunningSimulation;
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
  ];
  const relevantMutations = ["scenarioSettings/setScenario"];

  const isNamespaceMutation = namespaces.some((namespace) =>
    mutation.type.startsWith(namespace)
  );

  const isSpecificMutation = relevantMutations.includes(mutation.type);
  if (isNamespaceMutation || isSpecificMutation) {
    debouncedRenderLayout();
  }
});

export default store;
